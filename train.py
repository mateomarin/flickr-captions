import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import logging
from tqdm import tqdm
import wandb  # for experiment tracking

from image_captioning_model import ImageCaptioningModel
from dataset import Flickr30kDataset  # We'll need to create this

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: ImageCaptioningModel,
        train_dataset: Flickr30kDataset,
        val_dataset: Flickr30kDataset,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        warmup_steps: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        total_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.criterion = CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(project="image-captioning", entity="mateowilcke-mli")
            wandb.watch(model)

    # Training method - needs gradients
    def train_epoch(self, epoch: int):
        self.model.train()  # Set model to training mode
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            # batch contains: images (B, C, H, W) and captions (B, num_captions, seq_len)
            images, captions = batch
            batch_size, num_captions, seq_len = captions.shape
            
            # Repeat each image for its captions
            images = images.repeat_interleave(num_captions, dim=0)  # (B*num_captions, C, H, W)
            captions = captions.view(-1, seq_len)  # (B*num_captions, seq_len)
            
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Prepare inputs (shift right for teacher forcing)
            input_captions = captions[:, :-1]  # all but last token
            target_captions = captions[:, 1:]  # all but first token
            
            # Forward pass
            outputs = self.model(images, input_captions)
            
            # Calculate loss
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                target_captions.view(-1)
            )
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Update weights using the optimizer
                self.optimizer.step()
                
                # Update learning rate
                self.scheduler.step()
                
                # Zero out gradients for next step
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": total_loss / (step + 1),
                "lr": self.scheduler.get_last_lr()[0]
            })
            
            if self.use_wandb:
                wandb.log({
                    "train_loss": loss.item() * self.gradient_accumulation_steps,
                    "learning_rate": self.scheduler.get_last_lr()[0]
                })
        
        return total_loss / num_batches

    # Evaluation method - no gradients needed
    @torch.no_grad()  # Disable gradient computation
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        progress_bar = tqdm(self.val_loader, desc="Evaluating")
        
        for batch in progress_bar:
            images, captions = batch
            batch_size, num_captions, seq_len = captions.shape
            
            # Repeat each image for its captions
            images = images.repeat_interleave(num_captions, dim=0)
            captions = captions.view(-1, seq_len)
            
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            input_captions = captions[:, :-1]
            target_captions = captions[:, 1:]
            
            outputs = self.model(images, input_captions)
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                target_captions.view(-1)
            )
            
            total_loss += loss.item()
            
            progress_bar.set_postfix({"val_loss": total_loss / (num_batches)})
        
        return total_loss / num_batches

    def train(self):
        logger.info("Starting training...")
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()
            
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )
            
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss_epoch": train_loss,
                    "val_loss": val_loss
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    f"best_model.pth"
                )
                logger.info(f"Saved new best model with val_loss={val_loss:.4f}")

def main():
    # Initialize model and datasets
    model = ImageCaptioningModel()
    train_dataset = Flickr30kDataset(split="train")
    val_dataset = Flickr30kDataset(split="val")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 