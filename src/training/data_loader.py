"""
Multi-Domain Data Loader
Week 3-4: Router Training Infrastructure

Data loaders for training router with 3 domains:
- Domain 0: Code/Technical
- Domain 1: Communication
- Domain 2: Research/General
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import List, Dict, Optional
import json
import random


class DomainDataset(Dataset):
    """
    Dataset for a single domain with text samples.
    """

    def __init__(
        self,
        texts: List[str],
        domain_id: int,
        tokenizer,
        max_length: int = 512
    ):
        """
        Args:
            texts: List of text samples for this domain
            domain_id: Domain ID (0=code, 1=communication, 2=research)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.domain_id = domain_id
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a tokenized sample with domain label.

        Returns:
            dict with keys: input_ids, attention_mask, domain_label
        """
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'domain_label': torch.tensor(self.domain_id, dtype=torch.long)
        }


class MultiDomainDataLoader:
    """
    Data loader that handles multiple domains for router training.
    Supports balanced sampling across domains.
    """

    def __init__(
        self,
        domain_data: Dict[int, List[str]],
        tokenizer,
        batch_size: int = 16,
        max_length: int = 512,
        shuffle: bool = True,
        balanced_sampling: bool = True
    ):
        """
        Args:
            domain_data: Dict mapping domain_id -> list of text samples
                        {0: code_samples, 1: comm_samples, 2: research_samples}
            tokenizer: HuggingFace tokenizer
            batch_size: Batch size
            max_length: Maximum sequence length
            shuffle: Whether to shuffle data
            balanced_sampling: Whether to balance samples across domains
        """
        self.domain_data = domain_data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.balanced_sampling = balanced_sampling

        # Create individual domain datasets
        self.domain_datasets = {}
        for domain_id, texts in domain_data.items():
            self.domain_datasets[domain_id] = DomainDataset(
                texts=texts,
                domain_id=domain_id,
                tokenizer=tokenizer,
                max_length=max_length
            )

        # Create combined dataset
        if balanced_sampling:
            # Balance by sampling equal amounts from each domain
            self.dataset = self._create_balanced_dataset()
        else:
            # Just concatenate all domains
            self.dataset = ConcatDataset(list(self.domain_datasets.values()))

        # Create DataLoader
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )

    def _create_balanced_dataset(self):
        """
        Create balanced dataset by sampling equal amounts from each domain.
        Uses the size of the smallest domain.
        """
        # Find minimum domain size
        min_size = min(len(ds) for ds in self.domain_datasets.values())

        # Sample from each domain
        balanced_datasets = []
        for domain_id, dataset in self.domain_datasets.items():
            # Create subset with min_size samples
            if len(dataset) > min_size:
                indices = random.sample(range(len(dataset)), min_size)
                subset = torch.utils.data.Subset(dataset, indices)
                balanced_datasets.append(subset)
            else:
                balanced_datasets.append(dataset)

        return ConcatDataset(balanced_datasets)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'total_samples': len(self.dataset),
            'batch_size': self.batch_size,
            'num_batches': len(self),
            'domains': {}
        }

        for domain_id, dataset in self.domain_datasets.items():
            stats['domains'][domain_id] = {
                'samples': len(dataset),
                'name': self._get_domain_name(domain_id)
            }

        return stats

    @staticmethod
    def _get_domain_name(domain_id: int) -> str:
        """Get human-readable domain name"""
        names = {
            0: "Code/Technical",
            1: "Communication",
            2: "Research/General"
        }
        return names.get(domain_id, f"Domain {domain_id}")


def create_dummy_domain_data(
    num_samples_per_domain: int = 100,
    domain_names: Optional[List[str]] = None
) -> Dict[int, List[str]]:
    """
    Create dummy data for testing.

    Args:
        num_samples_per_domain: Number of samples per domain
        domain_names: Optional list of domain names for generating text

    Returns:
        domain_data: Dict mapping domain_id -> list of text samples
    """
    if domain_names is None:
        domain_names = ["code", "communication", "research"]

    domain_data = {}

    for domain_id, domain_name in enumerate(domain_names):
        samples = []
        for i in range(num_samples_per_domain):
            # Generate dummy text based on domain
            if domain_id == 0:  # Code
                text = f"def function_{i}(): return {i} + {i * 2}"
            elif domain_id == 1:  # Communication
                text = f"Dear colleague, this is email number {i} regarding the meeting."
            else:  # Research
                text = f"This research paper discusses topic {i} in depth."

            samples.append(text)

        domain_data[domain_id] = samples

    return domain_data


def test_data_loader():
    """Test multi-domain data loader"""
    print("=" * 60)
    print("Testing Multi-Domain Data Loader")
    print("=" * 60)

    # Create dummy tokenizer (using a simple one for testing)
    from transformers import AutoTokenizer

    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-1.2B")

    # Create dummy domain data
    print("\n2. Creating dummy domain data...")
    domain_data = create_dummy_domain_data(num_samples_per_domain=100)

    print(f"   Domain 0 (Code): {len(domain_data[0])} samples")
    print(f"   Domain 1 (Comm): {len(domain_data[1])} samples")
    print(f"   Domain 2 (Research): {len(domain_data[2])} samples")

    # Create data loader
    print("\n3. Creating data loader...")
    data_loader = MultiDomainDataLoader(
        domain_data=domain_data,
        tokenizer=tokenizer,
        batch_size=8,
        max_length=128,
        shuffle=True,
        balanced_sampling=True
    )

    # Print stats
    print("\n4. Data loader stats:")
    stats = data_loader.get_stats()
    for key, value in stats.items():
        if key != 'domains':
            print(f"   {key}: {value}")

    print(f"\n   Domain breakdown:")
    for domain_id, domain_stats in stats['domains'].items():
        print(f"     {domain_stats['name']}: {domain_stats['samples']} samples")

    # Test iteration
    print("\n5. Testing iteration...")
    batch = next(iter(data_loader))

    print(f"   Batch keys: {batch.keys()}")
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   attention_mask shape: {batch['attention_mask'].shape}")
    print(f"   domain_label shape: {batch['domain_label'].shape}")
    print(f"   Domain labels in batch: {batch['domain_label'].tolist()}")

    print("\n" + "=" * 60)
    print("✓ Data loader tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_data_loader()
