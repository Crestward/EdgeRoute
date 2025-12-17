"""
Router Loss Implementation
Week 3-4: Router Training Infrastructure

Implements loss function for training the domain router.
Combines classification loss with load balancing term.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RouterLoss(nn.Module):
    """
    Loss function for training domain router.
    Combines cross-entropy loss with load balancing regularization.
    """

    def __init__(
        self,
        num_experts: int = 3,
        load_balance_weight: float = 0.01,
        label_smoothing: float = 0.0
    ):
        """
        Args:
            num_experts: Number of experts (3 for code/communication/research)
            load_balance_weight: Weight for load balancing loss (default: 0.01)
            label_smoothing: Label smoothing factor for cross-entropy (default: 0.0)
        """
        super().__init__()
        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight

        # Cross-entropy loss with optional label smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        router_logits: torch.Tensor,
        expert_labels: torch.Tensor,
        compute_load_balance: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute router loss.

        Args:
            router_logits: [batch_size, seq_len, num_experts] or [batch_size, num_experts]
            expert_labels: [batch_size, seq_len] or [batch_size] - ground truth expert IDs
            compute_load_balance: Whether to add load balancing term

        Returns:
            loss: Total loss (scalar)
            loss_dict: Dictionary with loss components
        """
        # Flatten logits and labels if needed
        if router_logits.dim() == 3:
            # [batch, seq_len, num_experts] -> [batch * seq_len, num_experts]
            batch_size, seq_len, num_experts = router_logits.shape
            router_logits_flat = router_logits.reshape(-1, num_experts)
            expert_labels_flat = expert_labels.reshape(-1)
        else:
            router_logits_flat = router_logits
            expert_labels_flat = expert_labels

        # Classification loss (cross-entropy)
        ce_loss = self.ce_loss(router_logits_flat, expert_labels_flat)

        # Total loss starts with classification
        total_loss = ce_loss
        loss_dict = {
            'ce_loss': ce_loss.item(),
            'load_balance_loss': 0.0,
            'total_loss': ce_loss.item()
        }

        # Load balancing loss (encourage uniform expert usage)
        if compute_load_balance and self.load_balance_weight > 0:
            lb_loss = self._compute_load_balance_loss(router_logits_flat)
            total_loss = total_loss + self.load_balance_weight * lb_loss

            loss_dict['load_balance_loss'] = lb_loss.item()
            loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def _compute_load_balance_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage uniform expert usage.

        Uses auxiliary loss from Switch Transformer paper:
        L_aux = num_experts * sum(f_i * P_i)
        where f_i is fraction of tokens assigned to expert i
        and P_i is mean router probability for expert i

        Args:
            router_logits: [N, num_experts] where N = batch_size * seq_len

        Returns:
            load_balance_loss: Scalar loss
        """
        # Get router probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # [N, num_experts]

        # Get expert assignments (hard assignment)
        expert_indices = torch.argmax(router_logits, dim=-1)  # [N]

        # Compute fraction of tokens assigned to each expert (f_i)
        num_tokens = router_logits.shape[0]
        expert_fractions = torch.zeros(
            self.num_experts,
            device=router_logits.device,
            dtype=router_logits.dtype
        )

        for expert_id in range(self.num_experts):
            mask = (expert_indices == expert_id)
            expert_fractions[expert_id] = mask.float().sum() / num_tokens

        # Compute mean router probability for each expert (P_i)
        mean_router_probs = router_probs.mean(dim=0)  # [num_experts]

        # Load balance loss: encourages f_i * P_i to be uniform
        # Minimizing this encourages equal load across experts
        load_balance_loss = self.num_experts * (expert_fractions * mean_router_probs).sum()

        return load_balance_loss


class RouterMetrics:
    """
    Compute and track router training metrics.
    """

    def __init__(self, num_experts: int = 3):
        """
        Args:
            num_experts: Number of experts
        """
        self.num_experts = num_experts
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.total_correct = 0
        self.total_samples = 0
        self.expert_counts = [0] * self.num_experts
        self.expert_correct = [0] * self.num_experts

    def update(
        self,
        router_logits: torch.Tensor,
        expert_labels: torch.Tensor
    ):
        """
        Update metrics with batch results.

        Args:
            router_logits: [batch_size, num_experts] or [batch, seq_len, num_experts]
            expert_labels: [batch_size] or [batch, seq_len]
        """
        # Flatten if needed
        if router_logits.dim() == 3:
            router_logits = router_logits.reshape(-1, self.num_experts)
            expert_labels = expert_labels.reshape(-1)

        # Get predictions
        predictions = torch.argmax(router_logits, dim=-1)

        # Overall accuracy
        correct = (predictions == expert_labels)
        self.total_correct += correct.sum().item()
        self.total_samples += expert_labels.numel()

        # Per-expert accuracy
        for expert_id in range(self.num_experts):
            mask = (expert_labels == expert_id)
            if mask.any():
                self.expert_counts[expert_id] += mask.sum().item()
                self.expert_correct[expert_id] += (correct & mask).sum().item()

    def get_metrics(self) -> dict:
        """
        Get current metrics.

        Returns:
            metrics: Dictionary with accuracy metrics
        """
        metrics = {
            'accuracy': self.total_correct / max(self.total_samples, 1),
            'total_samples': self.total_samples
        }

        # Per-expert accuracy
        for expert_id in range(self.num_experts):
            if self.expert_counts[expert_id] > 0:
                acc = self.expert_correct[expert_id] / self.expert_counts[expert_id]
                metrics[f'expert_{expert_id}_accuracy'] = acc
                metrics[f'expert_{expert_id}_count'] = self.expert_counts[expert_id]
            else:
                metrics[f'expert_{expert_id}_accuracy'] = 0.0
                metrics[f'expert_{expert_id}_count'] = 0

        return metrics

    def print_metrics(self):
        """Print metrics in readable format"""
        metrics = self.get_metrics()

        print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({self.total_correct}/{self.total_samples})")
        print(f"\nPer-Expert Accuracy:")
        for expert_id in range(self.num_experts):
            acc = metrics[f'expert_{expert_id}_accuracy']
            count = metrics[f'expert_{expert_id}_count']
            print(f"  Expert {expert_id}: {acc:.4f} ({count} samples)")


def test_router_loss():
    """Test router loss implementation"""
    print("=" * 60)
    print("Testing Router Loss")
    print("=" * 60)

    # Configuration
    batch_size = 16
    seq_len = 10
    num_experts = 3

    # Create loss function
    print(f"\n1. Creating RouterLoss")
    router_loss_fn = RouterLoss(
        num_experts=num_experts,
        load_balance_weight=0.01,
        label_smoothing=0.1
    )
    print(f"   Num experts: {num_experts}")
    print(f"   Load balance weight: 0.01")
    print(f"   Label smoothing: 0.1")

    # Test with dummy data
    print(f"\n2. Testing loss computation")
    router_logits = torch.randn(batch_size, seq_len, num_experts)
    expert_labels = torch.randint(0, num_experts, (batch_size, seq_len))

    print(f"   Router logits shape: {router_logits.shape}")
    print(f"   Expert labels shape: {expert_labels.shape}")

    loss, loss_dict = router_loss_fn(router_logits, expert_labels)

    print(f"\n   Loss components:")
    for key, value in loss_dict.items():
        print(f"     {key}: {value:.4f}")

    print(f"   ✓ Loss computation successful")

    # Test metrics
    print(f"\n3. Testing RouterMetrics")
    metrics = RouterMetrics(num_experts=num_experts)

    # Update with random predictions
    for _ in range(5):
        router_logits = torch.randn(batch_size, num_experts)
        expert_labels = torch.randint(0, num_experts, (batch_size,))
        metrics.update(router_logits, expert_labels)

    print(f"\n   Metrics after 5 batches:")
    metrics.print_metrics()

    print("\n" + "=" * 60)
    print("✓ Router loss tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_router_loss()
