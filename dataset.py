import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from typing import List, Tuple
from datasets import load_dataset
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Flickr30kDataset(Dataset):
    def __init__(
        self,
        split: str = "test",
        transform = None,
        tokenizer = None,
        max_length: int = 30
    ):
        """
        Args:
            split (str): Currently only "test" is available in the HF dataset
            transform: Optional transform to be applied on images
            tokenizer: Tokenizer for processing captions
            max_length: Maximum length for caption tokenization
        """
        # Load dataset from Hugging Face
        logger.info(f"Loading Flickr30k {split} split...")
        try:
            self.dataset = load_dataset("nlphuji/flickr30k", split=split)
            logger.info(f"Loaded {len(self.dataset)} examples")
        except ValueError as e:
            logger.error(f"Error loading dataset: {e}")
            logger.info("Only 'test' split is available. Defaulting to test split.")
            self.dataset = load_dataset("nlphuji/flickr30k", split="test")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Detailed logging of first item
        if len(self.dataset) > 0:
            first_item = self.dataset[0]
            logger.info("\nDetailed first item structure:")
            logger.info(f"Keys in item: {first_item.keys()}")
            logger.info(f"Image type: {type(first_item['image'])}")
            logger.info(f"Image size: {first_item['image'].size}")
            logger.info(f"Number of captions: {len(first_item['caption'])}")
            logger.info("\nFirst few captions:")
            for i, cap in enumerate(first_item['caption'][:2]):
                logger.info(f"Caption {i+1}: {cap}")
            logger.info(f"Sentence IDs: {first_item['sentids']}")
            logger.info(f"Split info: {first_item['split']}")
            logger.info(f"Image ID: {first_item['img_id']}")
            logger.info(f"Filename: {first_item['filename']}")
            
            if self.tokenizer:
                logger.info("\nTokenizer info:")
                logger.info(f"Tokenizer type: {type(self.tokenizer)}")
                logger.info(f"Vocab size: {self.tokenizer.vocab_size}")
                logger.info(f"Max length: {self.max_length}")
                # Test tokenization on first caption
                test_tokens = self.tokenizer(
                    first_item['caption'][0],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                logger.info(f"Sample tokenized shape: {test_tokens.input_ids.shape}")
                logger.info(f"Sample token IDs: {test_tokens.input_ids[0][:10]}...")
        
        # Define image transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        logger.info(f"Using image transforms: {self.transform}")

        # Clean captions during dataset loading
        if len(self.dataset) > 0:
            cleaned_dataset = []
            for item in self.dataset:
                # Keep only the most concise caption for each image
                captions = item['caption']
                best_caption = min(captions, key=len)  # Use shortest caption
                
                # Create new item with single, clean caption
                cleaned_item = {
                    'image': item['image'],
                    'caption': best_caption,
                    'filename': item['filename']
                }
                cleaned_dataset.append(cleaned_item)
            
            self.dataset = cleaned_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Transformed image tensor (3, 224, 224)
            captions: Tokenized captions tensor (num_captions, max_length)
        """
        item = self.dataset[idx]
        
        # Load and transform image
        image = item['image']
        original_size = image.size
        image = self.transform(image)
        
        # Get captions and tokenize them
        captions = item['caption']
        
        # Detailed logging for first few items
        if idx < 3:
            logger.info(f"\nProcessing Dataset Item {idx}:")
            logger.info(f"  Filename: {item['filename']}")
            logger.info(f"  Original image size: {original_size}")
            logger.info(f"  Transformed image shape: {image.shape}")
            logger.info(f"  Number of raw captions: {len(captions)}")
            logger.info(f"  First caption: {captions[0]}")
            logger.info(f"  Image tensor range: [{image.min():.3f}, {image.max():.3f}]")
        
        if self.tokenizer:
            # Tokenize all captions for this image
            tokenized = self.tokenizer(
                captions,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                pad_to_max_length=True  # Ensure consistent padding
            )
            captions = tokenized.input_ids
            
            # Log tokenization details for first few items
            if idx < 3:
                logger.info(f"  Tokenized shape: {captions.shape}")
                logger.info(f"  First caption token IDs: {captions[0][:10]}...")
                logger.info(f"  Padding token ID: {self.tokenizer.pad_token_id}")
                # Only decode up to the first end token
                decoded = self.tokenizer.decode(captions[0])
                first_end = decoded.find('<|endoftext|>')
                if first_end != -1:
                    decoded = decoded[:first_end + len('<|endoftext|>')]
                logger.info(f"  First caption decoded: {decoded}")
        
        return image, captions

    def get_sample(self, idx: int = 0) -> None:
        """Debug method to examine a sample from the dataset"""
        image, captions = self[idx]
        logger.info("\nSample details:")
        logger.info(f"Image tensor shape: {image.shape}")
        logger.info(f"Image tensor range: [{image.min():.3f}, {image.max():.3f}]")
        logger.info(f"Number of captions: {len(captions)}")
        logger.info("Captions:")
        for i, cap in enumerate(captions, 1):
            logger.info(f"{i}. {cap}") 