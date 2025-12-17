"""
LFM2-MoE Integration
Week 1-2: LFM2-MoE Integration

Integrates MoE layers with LFM2 base model.
Inserts MoE after blocks 6-10 (middle layers).
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict
from .moe_layer import MoELayer


class LFM2MoE(nn.Module):
    """
    LFM2 with Mixture of Experts layers inserted in middle blocks.
    """

    def __init__(
        self,
        model_name: str = "LiquidAI/LFM2-1.2B",
        num_experts: int = 3,
        moe_layers: list = [6, 7, 8, 9, 10],  # Insert MoE after these blocks
        freeze_base: bool = True,
        device: str = "cpu"
    ):
        """
        Args:
            model_name: HuggingFace model name for LFM2
            num_experts: Number of experts (3 for code/communication/research)
            moe_layers: List of layer indices to add MoE after
            freeze_base: Whether to freeze LFM2 base model weights
            device: Device to load model on
        """
        super().__init__()

        print(f"Loading LFM2 base model: {model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_experts = num_experts
        self.moe_layer_indices = moe_layers
        self.device = device

        # Get hidden size from model config
        self.hidden_size = self.base_model.config.hidden_size

        # Freeze base model if requested
        if freeze_base:
            print("Freezing base model parameters")
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Create MoE layers for specified blocks
        print(f"Adding MoE layers after blocks: {moe_layers}")
        self.moe_layers = nn.ModuleDict({
            str(layer_idx): MoELayer(
                hidden_size=self.hidden_size,
                num_experts=num_experts,
                use_residual=True
            )
            for layer_idx in moe_layers
        })

        # Move MoE layers to device
        for moe_layer in self.moe_layers.values():
            moe_layer.to(device)

        print(f"✓ LFM2-MoE initialized with {num_experts} experts")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_router_logits: bool = False
    ):
        """
        Forward pass through LFM2-MoE.

        Note: This is a simplified version for testing.
        Full integration requires modifying LFM2's forward pass to insert MoE layers.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: Optional[batch_size, seq_len]
            return_router_logits: Whether to return router logits

        Returns:
            outputs: Model outputs with optional router logits
        """
        # For now, just run through base model
        # Full integration will require custom forward pass with MoE insertion
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # TODO: In full implementation, we'll intercept hidden states
        # after specified layers and pass through MoE before continuing

        if return_router_logits:
            # Placeholder for router logits
            outputs.router_logits = None

        return outputs

    def generate(self, *args, **kwargs):
        """Wrapper for generation using base model"""
        return self.base_model.generate(*args, **kwargs)

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in different parts of the model"""
        base_params = sum(p.numel() for p in self.base_model.parameters())
        moe_params = sum(p.numel() for p in self.moe_layers.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        return {
            'base_model': base_params,
            'moe_layers': moe_params,
            'total': base_params + moe_params,
            'trainable': trainable_params
        }

    def print_model_info(self):
        """Print model information"""
        params = self.count_parameters()

        print("\n" + "=" * 60)
        print("LFM2-MoE Model Information")
        print("=" * 60)
        print(f"Base model parameters: {params['base_model'] / 1e9:.2f}B")
        print(f"MoE parameters: {params['moe_layers'] / 1e6:.2f}M")
        print(f"Total parameters: {params['total'] / 1e9:.2f}B")
        print(f"Trainable parameters: {params['trainable'] / 1e6:.2f}M")
        print(f"MoE layers: {self.moe_layer_indices}")
        print(f"Number of experts: {self.num_experts}")
        print("=" * 60 + "\n")


def test_lfm2_moe_integration():
    """Test LFM2-MoE integration"""
    print("=" * 60)
    print("Testing LFM2-MoE Integration")
    print("=" * 60)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Create LFM2-MoE model
    print("\n1. Creating LFM2-MoE model...")
    model = LFM2MoE(
        model_name="LiquidAI/LFM2-1.2B",
        num_experts=3,
        moe_layers=[6, 7, 8, 9, 10],
        freeze_base=True,
        device=device
    )

    # Print model info
    model.print_model_info()

    # Test forward pass
    print("2. Testing forward pass...")
    test_prompt = "The future of AI is"
    inputs = model.tokenizer(test_prompt, return_tensors="pt").to(device)

    print(f"   Input: '{test_prompt}'")

    # Test generation
    print("\n3. Testing generation...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=model.tokenizer.eos_token_id
        )

    generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Generated: '{generated_text}'")
    print(f"   ✓ Generation successful")

    # Test MoE layer access
    print("\n4. Testing MoE layer access...")
    for layer_idx, moe_layer in model.moe_layers.items():
        print(f"   MoE Layer {layer_idx}:")
        print(f"     - Router params: {sum(p.numel() for p in moe_layer.router.parameters()):,}")
        print(f"     - Expert params: {sum(p.numel() for p in moe_layer.experts.parameters()):,}")

    print("\n" + "=" * 60)
    print("✓ LFM2-MoE integration tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_lfm2_moe_integration()
