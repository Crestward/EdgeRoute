"""
LoRA Expert Training Script for RunPod
Trains domain-specific LoRA adapters on GPU
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from datetime import datetime
import gc


def load_domain_data(data_dir, split="train"):
    """Load domain-specific training data"""
    filepath = os.path.join(data_dir, f"{split}.jsonl")
    samples = []

    print(f"Loading {split} data from: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            samples.append({'text': data['text']})

    print(f"Loaded {len(samples):,} samples")
    return Dataset.from_list(samples)


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize examples for causal LM"""
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding=False  # DataCollator will handle padding
    )
    # Don't add labels here - DataCollatorForLanguageModeling will handle it
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Train LoRA expert on RunPod")
    parser.add_argument("--expert_id", type=int, required=True, help="Expert ID (0/1/2)")
    parser.add_argument("--domain", required=True, help="Domain name (code/general/research)")
    parser.add_argument("--data_dir", required=True, help="Data directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=128, help="LoRA alpha")
    args = parser.parse_args()

    print("=" * 70)
    print(f"RUNPOD EXPERT TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"Expert ID: {args.expert_id}")
    print(f"Domain: {args.domain}")
    print(f"LoRA rank: {args.rank}, alpha: {args.alpha}")

    device = "cuda"
    print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load tokenizer and model
    print("\n1. Loading base model...")
    model_name = "LiquidAI/LFM2-1.2B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )

    # Apply LoRA
    print("\n2. Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print("\n3. Loading training data...")
    train_dataset = load_domain_data(args.data_dir, "train")
    val_dataset = load_domain_data(args.data_dir, "val")

    # Tokenize
    print("\n4. Tokenizing...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="epoch",  # Eval at end of each epoch
        save_strategy="steps",  # Save every N steps for crash protection
        save_steps=300,  # Save every 300 steps (~10-15 min on GPU)
        save_total_limit=3,  # Keep last 3 checkpoints
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        resume_from_checkpoint=True  # Auto-resume if checkpoint exists
    )

    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # Train
    print("\n5. Starting training...")
    print("=" * 70)

    trainer.train()

    # Save
    print("\n6. Saving model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save metadata
    metadata = {
        'expert_id': args.expert_id,
        'domain': args.domain,
        'rank': args.rank,
        'alpha': args.alpha,
        'epochs': args.epochs,
        'base_model': model_name,
        'final_loss': trainer.state.log_history[-1].get('loss', 'N/A'),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }

    with open(os.path.join(args.output_dir, 'expert_config.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Model saved to: {args.output_dir}")
    print(f"Adapter size: ~30-50 MB")

    # Clear memory
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
