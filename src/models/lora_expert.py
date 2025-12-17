"""
LoRA Expert Implementation
Week 5-6: Initial Expert Training

Implements LoRA (Low-Rank Adaptation) adapters for expert fine-tuning.
Uses rank=64, alpha=128 as specified in roadmap.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict
import os


class LoRAExpertConfig:
    """Configuration for LoRA expert adapters"""

    def __init__(
        self,
        expert_id: int,
        domain_name: str,
        rank: int = 64,
        alpha: int = 128,
        dropout: float = 0.1,
        target_modules: Optional[list] = None
    ):
        """
        Args:
            expert_id: Expert ID (0=code, 1=general, 2=research)
            domain_name: Domain name (code, general, research)
            rank: LoRA rank (default: 64)
            alpha: LoRA alpha (default: 128)
            dropout: LoRA dropout (default: 0.1)
            target_modules: Modules to apply LoRA to
        """
        self.expert_id = expert_id
        self.domain_name = domain_name
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        # Default target modules for LFM2 architecture
        if target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "up_proj", "down_proj",  # FFN
                "gate_proj"  # Gating
            ]
        else:
            self.target_modules = target_modules

    def to_peft_config(self) -> LoraConfig:
        """Convert to PEFT LoraConfig"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias="none",
            inference_mode=False
        )

    def __repr__(self):
        return (
            f"LoRAExpertConfig(expert_id={self.expert_id}, "
            f"domain={self.domain_name}, rank={self.rank}, "
            f"alpha={self.alpha}, dropout={self.dropout})"
        )


class LoRAExpert:
    """
    LoRA Expert wrapper for domain-specific fine-tuning.
    Manages a base model with LoRA adapters.
    """

    def __init__(
        self,
        base_model_name: str,
        expert_config: LoRAExpertConfig,
        device: str = "cpu"
    ):
        """
        Args:
            base_model_name: HuggingFace model name (e.g., "LiquidAI/LFM2-1.2B")
            expert_config: LoRA configuration
            device: Device to load model on
        """
        self.base_model_name = base_model_name
        self.expert_config = expert_config
        self.device = device

        print(f"Loading base model: {base_model_name}")
        print(f"Expert config: {expert_config}")

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.base_model.config.pad_token_id = self.tokenizer.eos_token_id

        # Apply LoRA
        peft_config = expert_config.to_peft_config()
        self.model = get_peft_model(self.base_model, peft_config)

        print(f"✓ LoRA expert created")
        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        """Print trainable parameter statistics"""
        trainable_params = 0
        all_params = 0

        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        trainable_pct = 100 * trainable_params / all_params

        print(f"\nTrainable parameters:")
        print(f"  All params:      {all_params:,} ({all_params/1e6:.2f}M)")
        print(f"  Trainable:       {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"  Trainable %:     {trainable_pct:.4f}%")

    def save_adapter(self, output_dir: str):
        """
        Save LoRA adapter weights (not full model).

        Args:
            output_dir: Directory to save adapter
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save adapter weights only (small - ~30-50MB)
        self.model.save_pretrained(output_dir)

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)

        # Save expert config
        import json
        config_path = os.path.join(output_dir, 'expert_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'expert_id': self.expert_config.expert_id,
                'domain_name': self.expert_config.domain_name,
                'rank': self.expert_config.rank,
                'alpha': self.expert_config.alpha,
                'dropout': self.expert_config.dropout,
                'target_modules': self.expert_config.target_modules,
                'base_model': self.base_model_name
            }, f, indent=2)

        print(f"✓ Adapter saved to: {output_dir}")

    @classmethod
    def load_adapter(
        cls,
        adapter_dir: str,
        base_model_name: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Load LoRA adapter from directory.

        Args:
            adapter_dir: Directory containing saved adapter
            base_model_name: Optional base model name (uses saved config if None)
            device: Device to load on

        Returns:
            LoRAExpert instance
        """
        import json

        # Load expert config
        config_path = os.path.join(adapter_dir, 'expert_config.json')
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        # Use saved base model if not provided
        if base_model_name is None:
            base_model_name = config_data['base_model']

        # Create expert config
        expert_config = LoRAExpertConfig(
            expert_id=config_data['expert_id'],
            domain_name=config_data['domain_name'],
            rank=config_data['rank'],
            alpha=config_data['alpha'],
            dropout=config_data['dropout'],
            target_modules=config_data.get('target_modules')
        )

        # Load base model
        from peft import PeftModel

        print(f"Loading adapter from: {adapter_dir}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )

        # Load adapter
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

        # Create expert instance manually
        expert = cls.__new__(cls)
        expert.base_model_name = base_model_name
        expert.expert_config = expert_config
        expert.device = device
        expert.base_model = base_model
        expert.model = model
        expert.tokenizer = tokenizer

        print(f"✓ Adapter loaded successfully")
        return expert

    def get_model(self):
        """Get the PEFT model for training"""
        return self.model

    def get_tokenizer(self):
        """Get tokenizer"""
        return self.tokenizer


def test_lora_expert():
    """Test LoRA expert implementation"""
    print("=" * 60)
    print("Testing LoRA Expert")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Create expert config
    print("\n1. Creating expert config...")
    config = LoRAExpertConfig(
        expert_id=0,
        domain_name="code",
        rank=64,
        alpha=128,
        dropout=0.1
    )
    print(f"✓ Config: {config}")

    # Create LoRA expert
    print("\n2. Creating LoRA expert...")
    expert = LoRAExpert(
        base_model_name="LiquidAI/LFM2-1.2B",
        expert_config=config,
        device=device
    )

    # Test generation
    print("\n3. Testing generation...")
    prompt = "def fibonacci(n):"
    inputs = expert.tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = expert.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=expert.tokenizer.eos_token_id
        )

    generated = expert.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Generated:\n{generated}")

    # Test save/load
    print("\n4. Testing save/load...")
    save_dir = "./test_adapter"
    expert.save_adapter(save_dir)

    print("\n5. Loading adapter...")
    loaded_expert = LoRAExpert.load_adapter(save_dir, device=device)
    print("✓ Adapter loaded successfully")

    # Clean up
    import shutil
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"✓ Cleaned up test directory")

    print("\n" + "=" * 60)
    print("✓ All LoRA expert tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_lora_expert()
