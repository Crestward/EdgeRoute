"""
Router Training Script for RunPod
Optimized for GPU training on RunPod pods
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import json
import argparse
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime

# Import from main codebase
from src.training import RouterLoss, RouterMetrics, RouterTrainer
from src.models import DomainRouter


class DomainDataset(Dataset):
    """Domain classification dataset"""

    def __init__(self, data_files, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for domain_id, filepath in data_files.items():
            print(f"Loading domain {domain_id}: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    self.samples.append({
                        'text': data['text'],
                        'domain_id': domain_id
                    })

        print(f"Total samples: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'domain_label': torch.tensor(sample['domain_id'], dtype=torch.long)
        }


class LFM2Router(nn.Module):
    """Router with LFM2 feature extractor"""

    def __init__(self, base_model_name="LiquidAI/LFM2-1.2B", num_experts=3, device="cuda"):
        super().__init__()

        self.base_model = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        hidden_size = self.base_model.config.hidden_size
        self.router_head = DomainRouter(hidden_size=hidden_size, num_experts=num_experts)

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        hidden_states = outputs.last_hidden_state

        # Mean pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            pooled = hidden_states.mean(dim=1)

        _, router_logits = self.router_head(pooled.unsqueeze(1))
        return router_logits.squeeze(1)


def main():
    parser = argparse.ArgumentParser(description="Train router on RunPod")
    parser.add_argument("--data_dir", required=True, help="Data directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    print("=" * 70)
    print(f"RUNPOD ROUTER TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    device = "cuda"

    # GPU info
    print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-1.2B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    train_files = {
        0: os.path.join(args.data_dir, "code/train.jsonl"),
        1: os.path.join(args.data_dir, "general/train.jsonl"),
        2: os.path.join(args.data_dir, "research/train.jsonl")
    }
    val_files = {
        0: os.path.join(args.data_dir, "code/val.jsonl"),
        1: os.path.join(args.data_dir, "general/val.jsonl"),
        2: os.path.join(args.data_dir, "research/val.jsonl")
    }

    print("\n2. Loading training data...")
    train_dataset = DomainDataset(train_files, tokenizer)
    val_dataset = DomainDataset(val_files, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Create model
    print("\n3. Creating router...")
    router = LFM2Router(device=device).to(device)

    trainable = sum(p.numel() for p in router.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,} ({trainable/1e6:.2f}M)")

    # Train
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, router.parameters()),
        lr=args.lr
    )
    loss_fn = RouterLoss(num_experts=3, load_balance_weight=0.01)

    trainer = RouterTrainer(
        router=router, optimizer=optimizer, loss_fn=loss_fn,
        device=device, use_amp=True, checkpoint_dir=args.output_dir
    )

    print("\n4. Starting training...")
    print("=" * 70)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        log_interval=100,
        save_steps=500  # Save every 500 steps (~10-15 min on GPU)
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best val accuracy: {max(history['val_accuracy']):.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
