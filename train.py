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
        
        # Add length penalty to loss
        self.length_penalty_weight = 0.1
        
        if use_wandb:
            wandb.init(project="image-captioning", entity="mateowilcke-mli")
            wandb.watch(model)

    # Training method - needs gradients
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            images, captions = batch
            batch_size, num_captions, seq_len = captions.shape
            
            # Repeat each image for its captions
            images = images.repeat_interleave(num_captions, dim=0)
            captions = captions.view(-1, seq_len)
            
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Prepare inputs (shift right for teacher forcing)
            input_captions = captions[:, :-1]
            target_captions = captions[:, 1:]
            
            # Forward pass
            outputs = self.model(images, input_captions)
            
            # Calculate loss with length penalty
            ce_loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                target_captions.reshape(-1)
            )
            
            # Add length penalty to encourage shorter captions
            caption_lengths = (target_captions != self.model.tokenizer.pad_token_id).sum(dim=1)
            length_penalty = self.length_penalty_weight * (caption_lengths.float() / caption_lengths.size(0))
            
            loss = ce_loss + length_penalty.mean()
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{total_loss / (step + 1):.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
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
                outputs.reshape(-1, outputs.size(-1)),
                target_captions.reshape(-1)
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
    logger.info("Initializing model and datasets...")
    model = ImageCaptioningModel()
    
    dataset = Flickr30kDataset(
        split="test",
        tokenizer=model.tokenizer
    )
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size]
    )
    
    logger.info(f"Dataset sizes:")
    logger.info(f"  Train: {len(train_dataset)}")
    logger.info(f"  Val: {len(val_dataset)}")
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    trainer.train()

if __name__ == "__main__":
    main() 