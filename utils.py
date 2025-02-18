import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def generate_modified_mask(sz, batch_size=None):
    """Create causal mask for transformer decoder.
    
    Args:
        sz: Sequence length
        batch_size: Optional batch size for broadcasting
        
    Returns:
        Mask of shape (batch_size * num_heads, sz, sz) for MultiheadAttention
    """
    # Create causal mask (upper triangular matrix)
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    # Convert to float mask with -inf for blocked attention
    mask = mask.float().masked_fill(mask == True, float('-inf'))
    
    if batch_size is not None:
        # For MultiheadAttention with batch_first=True
        # We need to expand for (batch_size * num_heads, seq_len, seq_len)
        num_heads = 8  # This should match nhead in the model
        mask = mask.unsqueeze(0).expand(batch_size * num_heads, -1, -1)
    
    return mask

def process_captions(tokenizer, captions, device, max_length):
    """Process captions and return token ids and attention mask."""
    if isinstance(captions[0], str):
        tokenized = tokenizer(
            captions,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        captions = tokenized.input_ids.to(device)
        # Only create attention_mask if we have variable length sequences
        attention_mask = None
        if tokenized.attention_mask.any() != tokenized.attention_mask.all():
            attention_mask = tokenized.attention_mask.to(device)
    else:
        # Only create mask if there's actual padding
        if (captions != tokenizer.pad_token_id).any() != (captions != tokenizer.pad_token_id).all():
            attention_mask = (captions != tokenizer.pad_token_id)
        else:
            attention_mask = None
    
    return captions, attention_mask

def extend_padding_mask(padding_mask, batch_size, device):
    """Extend padding mask to include image token."""
    if padding_mask is None:
        return None
    
    # Create mask for image token (always valid)
    image_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
    
    # Concatenate with caption padding mask
    extended_mask = torch.cat([image_mask, padding_mask], dim=1)
    
    return extended_mask 