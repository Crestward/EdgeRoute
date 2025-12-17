"""
Router Training Loop
Week 3-4: Router Training Infrastructure

Complete training infrastructure for domain router with:
- Mixed precision training (BFloat16)
- Checkpointing (save/resume)
- Progress tracking
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict
import os
import json
from tqdm import tqdm

from .router_loss import RouterLoss, RouterMetrics


class RouterTrainer:
    """
    Trainer for domain router with mixed precision and checkpointing.
    """

    def __init__(
        self,
        router: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: RouterLoss,
        device: str = "cpu",
        use_amp: bool = False,
        checkpoint_dir: str = "./checkpoints"
    ):
        """
        Args:
            router: Router model to train
            optimizer: Optimizer
            loss_fn: RouterLoss instance
            device: Device for training
            use_amp: Use automatic mixed precision (BFloat16)
            checkpoint_dir: Directory to save checkpoints
        """
        self.router = router
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.use_amp = use_amp
        self.checkpoint_dir = checkpoint_dir

        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp and device == "cuda" else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(
        self,
        train_loader,
        epoch: int,
        log_interval: int = 10,
        save_steps: int = 500
    ) -> Dict:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            log_interval: How often to log (in batches)

        Returns:
            epoch_metrics: Dictionary with training metrics
        """
        self.router.train()
        metrics = RouterMetrics(num_experts=self.loss_fn.num_experts)

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            domain_labels = batch['domain_label'].to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(dtype=torch.bfloat16):
                    router_logits = self.router(input_ids)
                    loss, loss_dict = self.loss_fn(router_logits, domain_labels)
            else:
                router_logits = self.router(input_ids)
                loss, loss_dict = self.loss_fn(router_logits, domain_labels)

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Update metrics
            with torch.no_grad():
                metrics.update(router_logits, domain_labels)

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Save checkpoint every N steps
            if save_steps > 0 and self.global_step % save_steps == 0:
                avg_loss = total_loss / num_batches
                print(f"\n[Step {self.global_step}] Saving checkpoint (loss: {avg_loss:.4f})...")
                self.save_checkpoint(
                    epoch=epoch,
                    val_loss=avg_loss,
                    is_best=False,
                    extra_state={'step_checkpoint': True}
                )

            # Update progress bar
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{metrics.get_metrics()["accuracy"]:.4f}'
                })

        # Get final metrics
        epoch_metrics = metrics.get_metrics()
        epoch_metrics['loss'] = total_loss / num_batches

        return epoch_metrics

    @torch.no_grad()
    def validate(self, val_loader) -> Dict:
        """
        Validate the router.

        Args:
            val_loader: Validation data loader

        Returns:
            val_metrics: Validation metrics
        """
        self.router.eval()
        metrics = RouterMetrics(num_experts=self.loss_fn.num_experts)

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            domain_labels = batch['domain_label'].to(self.device)

            # Forward pass
            if self.use_amp:
                with autocast(dtype=torch.bfloat16):
                    router_logits = self.router(input_ids)
                    loss, _ = self.loss_fn(router_logits, domain_labels)
            else:
                router_logits = self.router(input_ids)
                loss, _ = self.loss_fn(router_logits, domain_labels)

            # Update metrics
            metrics.update(router_logits, domain_labels)
            total_loss += loss.item()
            num_batches += 1

        # Get final metrics
        val_metrics = metrics.get_metrics()
        val_metrics['loss'] = total_loss / num_batches

        return val_metrics

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
        extra_state: Optional[Dict] = None
    ):
        """
        Save training checkpoint.

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
            extra_state: Optional extra state to save
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'router_state_dict': self.router.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if extra_state is not None:
            checkpoint.update(extra_state)

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_router.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            checkpoint: Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.router.load_state_dict(checkpoint['router_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")

        return checkpoint

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        log_interval: int = 10,
        save_interval: int = 1,
        save_steps: int = 500
    ) -> Dict:
        """
        Complete training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            log_interval: Log every N batches
            save_interval: Save checkpoint every N epochs
            save_steps: Save checkpoint every N steps (0 to disable)

        Returns:
            training_history: Dictionary with training history
        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        print("=" * 60)
        print("Starting Router Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Num epochs: {num_epochs}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")

        # Check for existing checkpoint and auto-resume
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        if os.path.exists(checkpoint_path):
            print(f"\n✓ Found existing checkpoint: {checkpoint_path}")
            print("  Automatically resuming training...")
            self.load_checkpoint(checkpoint_path)

        print("=" * 60)

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader, epoch, log_interval, save_steps)

            print(f"\nEpoch {epoch} Training Results:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Accuracy: {train_metrics['accuracy']:.4f}")

            # Validate
            val_metrics = self.validate(val_loader)

            print(f"\nEpoch {epoch} Validation Results:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")

            # Save history
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])

            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch, val_metrics['loss'], is_best, history)

            print("-" * 60)

        # Save final training history
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)

        return history


def test_router_trainer():
    """Test router trainer with dummy data"""
    print("=" * 60)
    print("Testing Router Trainer")
    print("=" * 60)

    from transformers import AutoTokenizer
    from ..models import DomainRouter
    from .data_loader import MultiDomainDataLoader, create_dummy_domain_data

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = 2048
    num_experts = 3

    print(f"\nDevice: {device}")

    # Create dummy data
    print("\n1. Creating dummy data...")
    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-1.2B")
    domain_data = create_dummy_domain_data(num_samples_per_domain=100)

    train_loader = MultiDomainDataLoader(
        domain_data=domain_data,
        tokenizer=tokenizer,
        batch_size=8,
        max_length=128,
        balanced_sampling=True
    )

    val_data = create_dummy_domain_data(num_samples_per_domain=30)
    val_loader = MultiDomainDataLoader(
        domain_data=val_data,
        tokenizer=tokenizer,
        batch_size=8,
        max_length=128,
        balanced_sampling=True
    )

    # Create simple router (not using full LFM2 for testing)
    print("\n2. Creating router...")
    class SimpleRouter(nn.Module):
        def __init__(self, vocab_size, hidden_size, num_experts):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.router = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, num_experts)
            )

        def forward(self, input_ids):
            x = self.embedding(input_ids).mean(dim=1)  # Simple averaging
            return self.router(x)

    router = SimpleRouter(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_experts=num_experts
    ).to(device)

    # Create optimizer and loss
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-4)
    loss_fn = RouterLoss(num_experts=num_experts, load_balance_weight=0.01)

    # Create trainer
    print("\n3. Creating trainer...")
    trainer = RouterTrainer(
        router=router,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        use_amp=(device == "cuda"),  # Use AMP only on GPU
        checkpoint_dir="./test_checkpoints"
    )

    # Train for 2 epochs
    print("\n4. Training for 2 epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
        log_interval=5,
        save_interval=1
    )

    print("\n5. Training history:")
    for key, values in history.items():
        print(f"   {key}: {values}")

    print("\n" + "=" * 60)
    print("✓ Router trainer tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_router_trainer()
