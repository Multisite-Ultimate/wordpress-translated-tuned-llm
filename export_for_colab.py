#!/usr/bin/env python3
"""Export a subset of the training dataset for Google Colab.

This script creates a smaller JSONL file that can be uploaded to Colab
for training on the free tier.

Usage:
    python export_for_colab.py [--samples 10000] [--output train_subset.jsonl]
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser(description="Export dataset subset for Colab")
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of samples to export (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="train_subset.jsonl",
        help="Output filename (default: train_subset.jsonl)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/datasets/nl",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Run './run.py dataset nl' first to create the dataset.")
        return 1

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(str(dataset_path))

    # Get train split
    if "train" in dataset:
        train_data = dataset["train"]
    else:
        train_data = dataset

    total_samples = len(train_data)
    print(f"Total training samples: {total_samples:,}")

    # Sample subset
    num_samples = min(args.samples, total_samples)
    print(f"Sampling {num_samples:,} examples...")

    random.seed(args.seed)
    indices = random.sample(range(total_samples), num_samples)

    # Export to JSONL
    output_path = Path(args.output)
    print(f"Exporting to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, idx in enumerate(indices):
            example = train_data[idx]
            # Only keep the 'text' field for training
            record = {"text": example["text"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 1000 == 0:
                print(f"  Exported {i + 1:,} / {num_samples:,}")

    # File size
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\nExport complete!")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size:.1f} MB")
    print(f"  Samples: {num_samples:,}")

    print(f"\nNext steps:")
    print(f"  1. Open Google Colab: https://colab.research.google.com/")
    print(f"  2. Upload 'colab_training.ipynb' to Colab")
    print(f"  3. Set runtime to GPU (Runtime > Change runtime type > T4 GPU)")
    print(f"  4. Run the notebook and upload '{output_path}' when prompted")

    return 0


if __name__ == "__main__":
    exit(main())
