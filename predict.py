import torch
from PIL import Image
import logging
from image_captioning_model import ImageCaptioningModel
from torchvision import transforms
import argparse
from pathlib import Path
from typing import List
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image(image_path: str, device: str = "cuda"):
    """Load and preprocess an image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    return image

def generate_caption(
    model: ImageCaptioningModel,
    image: torch.Tensor,
    max_length: int = 30,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    min_length: int = 5,
    repetition_penalty: float = 1.2,
    length_penalty: float = 1.0
):
    """Generate a caption for the image using nucleus sampling."""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Get image features
        image_features = model.clip.get_image_features(image)
        
        # Initialize caption with start token
        caption = torch.tensor([[model.tokenizer.bos_token_id]], device=device)
        
        # Keep track of generated tokens for repetition penalty
        generated_tokens = set()
        
        # Generate tokens one by one
        for _ in range(max_length):
            # Get text embeddings for current caption
            text_embeddings = model.text_embedding(caption)
            
            # Combine with image features
            sequence = torch.cat([
                image_features.unsqueeze(1),
                text_embeddings
            ], dim=1)
            
            # Add positional encoding
            sequence = model.positional_encoding(sequence)
            
            # Create causal mask
            seq_length = sequence.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_length, seq_length, device=device) * float('-inf'),
                diagonal=1
            )
            
            # Get decoder output
            decoder_output = model.decoder(sequence, mask=causal_mask)
            
            # Get next token logits
            next_token_logits = model.fc_out(decoder_output[:, -1, :])
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            for token_id in generated_tokens:
                next_token_logits[0, token_id] /= repetition_penalty
            
            # Apply length penalty
            if caption.size(1) > min_length:
                next_token_logits = next_token_logits / (caption.size(1) ** length_penalty)
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated tokens set
            generated_tokens.add(next_token.item())
            
            # Append next token to caption
            caption = torch.cat([caption, next_token], dim=1)
            
            # Stop if we generate end token or period
            if (next_token.item() == model.tokenizer.eos_token_id or 
                next_token.item() == model.tokenizer.encode('.')[0]):
                break
    
    # Decode caption and clean up
    caption = model.tokenizer.decode(caption[0], skip_special_tokens=True)
    
    # Clean up the caption
    caption = caption.strip()
    if not caption.endswith('.'):
        caption += '.'
    
    # Remove any trailing fragments after the last period
    last_period = caption.rfind('.')
    if last_period > 0:
        caption = caption[:last_period + 1]
    
    return caption

def get_original_captions(filename: str) -> List[str]:
    """Get original captions for an image from Flickr30k dataset."""
    try:
        dataset = load_dataset("nlphuji/flickr30k", split="test")
        # Find the image by filename
        for item in dataset:
            if item['filename'] == filename:
                return item['caption']
    except Exception as e:
        logger.warning(f"Could not fetch original captions: {e}")
    return []

def get_random_sample():
    """Get a random image and its captions from Flickr30k dataset."""
    try:
        dataset = load_dataset("nlphuji/flickr30k", split="test")
        idx = torch.randint(len(dataset), (1,)).item()
        sample = dataset[idx]
        return sample['image'], sample['caption'], sample['filename']
    except Exception as e:
        logger.error(f"Could not load random sample: {e}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Generate captions for images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--image_path", type=str, help="Path to image file or directory (optional)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) filtering parameter")
    parser.add_argument("--max_length", type=int, default=30, help="Maximum caption length")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Penalty for token repetition")
    parser.add_argument("--show_original", action="store_true", help="Show original captions from dataset")
    args = parser.parse_args()
    
    # Initialize model
    logger.info("Loading model...")
    model = ImageCaptioningModel()
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model = model.to(args.device)
    model.eval()
    
    if args.image_path is None:
        # Use random image from Flickr30k
        logger.info("Loading random image from Flickr30k dataset...")
        image, captions, filename = get_random_sample()
        if image is None:
            logger.error("Failed to load random sample")
            return
        
        # Transform image for model
        transformed_image = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])(image).unsqueeze(0).to(args.device)
        
        # Generate caption
        generated_caption = generate_caption(
            model, transformed_image,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        logger.info(f"\nImage: {filename}")
        logger.info(f"Generated caption: {generated_caption}")
        logger.info("\nOriginal captions:")
        for i, cap in enumerate(captions, 1):
            logger.info(f"{i}. {cap}")
    
    else:
        # Process provided image(s)
        image_path = Path(args.image_path)
        if image_path.is_file():
            # Single image
            image = load_image(str(image_path), args.device)
            caption = generate_caption(
                model, image,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            logger.info(f"\nImage: {image_path.name}")
            logger.info(f"Generated caption: {caption}")
            
            if args.show_original:
                original_captions = get_original_captions(image_path.name)
                if original_captions:
                    logger.info("\nOriginal captions:")
                    for i, cap in enumerate(original_captions, 1):
                        logger.info(f"{i}. {cap}")
                else:
                    logger.info("No original captions found in dataset")
        else:
            # Directory of images
            for img_file in image_path.glob("*.jpg"):
                image = load_image(str(img_file), args.device)
                caption = generate_caption(
                    model, image,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                logger.info(f"\nImage: {img_file.name}")
                logger.info(f"Generated caption: {caption}")
                
                if args.show_original:
                    original_captions = get_original_captions(img_file.name)
                    if original_captions:
                        logger.info("\nOriginal captions:")
                        for i, cap in enumerate(original_captions, 1):
                            logger.info(f"{i}. {cap}")
                    else:
                        logger.info("No original captions found in dataset")

if __name__ == "__main__":
    main() 