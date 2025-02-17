import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer
from .utils import PositionalEncoding, generate_modified_mask, process_captions

class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None, padding_mask=None):
        output = x
        
        for layer in self.layers:
            output = layer(output, mask, padding_mask)
            
        return self.norm(output)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        
        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, padding_mask=None):
        # Self-attention block
        _x = x
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=mask,
            key_padding_mask=padding_mask
        )[0]
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # Feedforward block
        _x = x
        x = self.ff(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        
        return x 
    
class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        max_length=77,  # CLIP's default max length
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    ):
        super().__init__()
        
        # CLIP Image Encoder and Tokenizer
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # Project image features to match transformer dimensions
        self.image_projection = nn.Linear(
            self.clip.config.vision_config.hidden_size,
            d_model
        )
        
        # Text embedding layer (using CLIP's vocab size)
        vocab_size = self.tokenizer.vocab_size
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Simplified Decoder
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_length = max_length

    def forward(self, images, captions):
        # Get image features and project them
        image_features = self.clip.get_image_features(images)
        image_features = self.image_projection(image_features)
        
        # Process captions and get attention mask
        captions, attention_mask = process_captions(self.tokenizer, captions, images.device, self.max_length)
        
        # Get text embeddings
        text_embeddings = self.text_embedding(captions)
        
        # Combine image features with text embeddings
        sequence = torch.cat([
            image_features.unsqueeze(1),
            text_embeddings
        ], dim=1)
        
        # Add positional encoding
        sequence = self.positional_encoding(sequence)
        
        # Create causal mask
        seq_length = sequence.size(1)
        causal_mask = generate_modified_mask(seq_length).to(sequence.device)
        
        # Pass through decoder
        decoder_output = self.decoder(
            sequence,
            mask=causal_mask,
            padding_mask=~attention_mask if attention_mask is not None else None
        )
        
        # Remove image token from output and project to vocabulary
        output = self.fc_out(decoder_output[:, 1:])
        return output