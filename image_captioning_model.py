import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer
from utils import PositionalEncoding, generate_modified_mask, process_captions, extend_padding_mask
import logging

logger = logging.getLogger(__name__)

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
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
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
            key_padding_mask=padding_mask,
            need_weights=False
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
        max_length=77,
        d_model=512,  # This matches CLIP's projected dimension
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
            
        # Add logging to check CLIP's output dimension
        logger.info(f"\nCLIP Model Info:")
        logger.info(f"Vision embedding dim: {self.clip.config.vision_config.hidden_size}")
        logger.info(f"Text embedding dim: {self.clip.config.text_config.hidden_size}")
        
        # No need for projection since CLIP already outputs 512-dim features
        # self.image_projection = nn.Linear(
        #     self.clip.config.vision_config.hidden_size,
        #     d_model
        # )
        
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
        self.nhead = nhead  # Store number of heads
        
        # Add a length controller
        self.length_controller = nn.Linear(d_model, 1)

    def forward(self, images, captions):
        # Get image features (already in correct dimension)
        image_features = self.clip.get_image_features(images)
        
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
        
        # Create causal mask with batch size and number of heads
        seq_length = sequence.size(1)
        batch_size = sequence.size(0)
        causal_mask = generate_modified_mask(seq_length, batch_size).to(sequence.device)
        
        # Extend padding mask to include image token
        if attention_mask is not None:
            attention_mask = extend_padding_mask(attention_mask, batch_size, images.device)
        
        # Pass through decoder
        decoder_output = self.decoder(
            sequence,
            mask=causal_mask,
            padding_mask=~attention_mask if attention_mask is not None else None
        )
        
        # Get output logits
        output = self.fc_out(decoder_output[:, 1:])
        
        # Add length control (matching dimensions)
        length_logits = self.length_controller(decoder_output[:, 1:])  # Match the output size
        length_weights = torch.sigmoid(length_logits)
        output = output * length_weights  # Broadcasting will handle the last dimension
        
        return output