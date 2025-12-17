"""
MoE Layer Implementation for LFM2
Week 1-2: LFM2-MoE Integration

Implements a Mixture of Experts layer that can be inserted after LFM2 blocks 6-10.
Uses top-1 routing for cache efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DomainRouter(nn.Module):
    """
    Router that selects which expert to use for each token.
    Uses top-1 routing for simplicity and cache efficiency.
    """

    def __init__(self, hidden_size: int, num_experts: int = 3):
        """
        Args:
            hidden_size: Size of hidden states from LFM2 (2048)
            num_experts: Number of experts (default: 3)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        # Router network: hidden_size -> num_experts
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_experts)
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            expert_indices: [batch_size, seq_len] - which expert to use for each token
            router_logits: [batch_size, seq_len, num_experts] - raw router scores
        """
        # Get router logits
        router_logits = self.router(hidden_states)  # [batch, seq_len, num_experts]

        # Top-1 routing: select expert with highest score
        expert_indices = torch.argmax(router_logits, dim=-1)  # [batch, seq_len]

        return expert_indices, router_logits


class DummyExpert(nn.Module):
    """
    Dummy expert for testing (random weights).
    In production, these will be LoRA adapters or fine-tuned layers.
    """

    def __init__(self, hidden_size: int, expert_id: int = 0):
        """
        Args:
            hidden_size: Size of hidden states (2048 for LFM2)
            expert_id: Identifier for this expert (0, 1, 2)
        """
        super().__init__()
        self.expert_id = expert_id
        self.hidden_size = hidden_size

        # Simple feedforward network
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        return self.net(hidden_states)


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with top-1 routing.
    Can be inserted after LFM2 blocks for domain specialization.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_experts: int = 3,
        use_residual: bool = True
    ):
        """
        Args:
            hidden_size: Size of hidden states from LFM2
            num_experts: Number of experts (default: 3 for code/communication/research)
            use_residual: Whether to add residual connection
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.use_residual = use_residual

        # Router
        self.router = DomainRouter(hidden_size, num_experts)

        # Experts (dummy for now, will be replaced with LoRA adapters)
        self.experts = nn.ModuleList([
            DummyExpert(hidden_size, expert_id=i)
            for i in range(num_experts)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_router_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MoE layer.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            return_router_logits: Whether to return router logits for training

        Returns:
            output: [batch_size, seq_len, hidden_size]
            router_logits: Optional[batch_size, seq_len, num_experts]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Route tokens to experts
        expert_indices, router_logits = self.router(hidden_states)

        # Process tokens through selected experts
        # For top-1 routing, we process each expert's tokens separately
        output = torch.zeros_like(hidden_states)

        for expert_id in range(self.num_experts):
            # Mask for tokens assigned to this expert
            expert_mask = (expert_indices == expert_id)  # [batch, seq_len]

            if expert_mask.any():
                # Get tokens for this expert
                # We process all positions but will only keep the masked ones
                expert_output = self.experts[expert_id](hidden_states)

                # Apply mask and add to output
                expert_mask_expanded = expert_mask.unsqueeze(-1).expand_as(hidden_states)
                output = torch.where(expert_mask_expanded, expert_output, output)

        # Optional residual connection
        if self.use_residual:
            output = output + hidden_states

        if return_router_logits:
            return output, router_logits
        else:
            return output, None

    def get_expert_usage_stats(self, expert_indices: torch.Tensor) -> dict:
        """
        Calculate expert usage statistics for monitoring load balancing.

        Args:
            expert_indices: [batch_size, seq_len]

        Returns:
            stats: Dict with expert usage percentages
        """
        total_tokens = expert_indices.numel()
        stats = {}

        for expert_id in range(self.num_experts):
            count = (expert_indices == expert_id).sum().item()
            percentage = (count / total_tokens) * 100
            stats[f'expert_{expert_id}'] = percentage

        return stats


def test_moe_layer():
    """Test MoE layer implementation"""
    print("=" * 60)
    print("Testing MoE Layer")
    print("=" * 60)

    # LFM2 configuration
    hidden_size = 2048
    num_experts = 3
    batch_size = 2
    seq_len = 10

    # Create MoE layer
    print(f"\n1. Creating MoE layer")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Number of experts: {num_experts}")

    moe_layer = MoELayer(hidden_size=hidden_size, num_experts=num_experts)

    # Count parameters
    total_params = sum(p.numel() for p in moe_layer.parameters())
    router_params = sum(p.numel() for p in moe_layer.router.parameters())
    expert_params = sum(p.numel() for p in moe_layer.experts.parameters())

    print(f"   Total parameters: {total_params:,}")
    print(f"   Router parameters: {router_params:,}")
    print(f"   Expert parameters: {expert_params:,}")

    # Test forward pass
    print(f"\n2. Testing forward pass")
    dummy_input = torch.randn(batch_size, seq_len, hidden_size)
    print(f"   Input shape: {dummy_input.shape}")

    output, router_logits = moe_layer(dummy_input, return_router_logits=True)

    print(f"   Output shape: {output.shape}")
    print(f"   Router logits shape: {router_logits.shape}")
    print(f"   ✓ Forward pass successful")

    # Test expert routing
    print(f"\n3. Testing expert routing")
    expert_indices, _ = moe_layer.router(dummy_input)
    stats = moe_layer.get_expert_usage_stats(expert_indices)

    print(f"   Expert usage:")
    for expert_name, percentage in stats.items():
        print(f"     {expert_name}: {percentage:.1f}%")

    # Test memory footprint
    print(f"\n4. Memory footprint")
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
    print(f"   Parameter memory: {param_memory:.2f} MB")

    print("\n" + "=" * 60)
    print("✓ MoE layer tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_moe_layer()
