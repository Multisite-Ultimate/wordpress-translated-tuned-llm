#!/usr/bin/env python3
"""Upload WordPress translation dataset to Hugging Face Hub."""

import argparse
import json
from pathlib import Path

from datasets import load_from_disk, DatasetDict
from huggingface_hub import HfApi, create_repo, login


def create_dataset_card(locale: str, stats_path: Path) -> str:
    """Create a README.md dataset card."""
    stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

    dataset_stats = stats.get("dataset_stats", {})

    return f"""---
language:
  - en
  - {locale}
license: apache-2.0
task_categories:
  - translation
tags:
  - wordpress
  - translation
  - fine-tuning
  - mistral
size_categories:
  - 100K<n<1M
---

# WordPress Translation Dataset (English → {locale.upper()})

Translation pairs extracted from WordPress plugins, themes, and core translations.
Designed for fine-tuning LLMs on WordPress-specific translation tasks.

## Dataset Description

This dataset contains English to {locale.upper()} translation pairs extracted from
the official WordPress translation project (translate.wordpress.org).

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Training examples | {dataset_stats.get('num_train_examples', 'N/A'):,} |
| Test examples | {dataset_stats.get('num_test_examples', 'N/A'):,} |
| Avg source length | {dataset_stats.get('avg_source_length', 'N/A')} chars |
| Avg target length | {dataset_stats.get('avg_target_length', 'N/A')} chars |

### Source Distribution

The translations come from:
- **WordPress Plugins** (~80%)
- **WordPress Themes** (~13%)
- **WordPress Core** (~7%)

## Dataset Structure

Each example contains:

| Field | Description |
|-------|-------------|
| `text` | Full training example with Mistral instruction format |
| `prompt` | Inference prompt (without target) |
| `source` | Original English text |
| `target` | {locale.upper()} translation |
| `project_type` | Source type (wp-plugins, wp-themes, etc.) |
| `project_name` | Original project name |

### Example

```python
from datasets import load_dataset

dataset = load_dataset("YOUR_USERNAME/wordpress-translations-{locale}")

print(dataset["train"][0])
# {{
#   "source": "Add to cart",
#   "target": "Toevoegen aan winkelwagen",
#   "text": "<s>[INST] Translate... [/INST]Toevoegen aan winkelwagen</s>",
#   ...
# }}
```

## Intended Use

This dataset is designed for:
- Fine-tuning LLMs for WordPress translation
- Training specialized translation models
- Evaluating translation quality on WordPress terminology

## Training with AutoTrain

1. Go to [AutoTrain](https://huggingface.co/autotrain)
2. Select "LLM Fine-tuning"
3. Choose this dataset
4. Select base model: `mistralai/Mistral-7B-Instruct-v0.2`
5. Configure training:
   - Text column: `text`
   - Train split: `train`
   - Validation split: `test`

## License

Apache 2.0 - Same as WordPress translations.

## Citation

If you use this dataset, please cite:

```
@dataset{{wordpress_translations_{locale},
  title={{WordPress Translation Dataset (EN-{locale.upper()})}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/YOUR_USERNAME/wordpress-translations-{locale}}}
}}
```
"""


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument("locale", help="Locale code (e.g., 'nl', 'de')")
    parser.add_argument(
        "--repo-name",
        help="Repository name (default: wordpress-translations-{locale})",
    )
    parser.add_argument(
        "--organization",
        help="Organization name (optional, uses your username if not specified)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/datasets"),
        help="Local datasets directory",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    # Login to Hugging Face
    if args.token:
        login(token=args.token)
    else:
        print("Please login to Hugging Face Hub.")
        print("You can either:")
        print("  1. Run 'huggingface-cli login' first")
        print("  2. Set HF_TOKEN environment variable")
        print("  3. Use --token argument")
        print()
        login()

    # Load local dataset
    dataset_path = args.data_dir / args.locale
    stats_path = dataset_path / "stats.json"

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Run './run.py dataset nl' first to build the dataset.")
        return 1

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(str(dataset_path))
    print(f"  Train: {len(dataset['train']):,} examples")
    print(f"  Test: {len(dataset['test']):,} examples")

    # Determine repo name
    repo_name = args.repo_name or f"wordpress-translations-{args.locale}"
    if args.organization:
        repo_id = f"{args.organization}/{repo_name}"
    else:
        api = HfApi()
        user_info = api.whoami()
        repo_id = f"{user_info['name']}/{repo_name}"

    print(f"\nUploading to: https://huggingface.co/datasets/{repo_id}")
    print(f"Visibility: {'private' if args.private else 'public'}")

    # Create the dataset card
    readme_content = create_dataset_card(args.locale, stats_path)

    # Push to Hub
    print("\nPushing dataset to Hub...")
    dataset.push_to_hub(
        repo_id,
        private=args.private,
    )

    # Update README
    print("Updating dataset card...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"\n✓ Dataset uploaded successfully!")
    print(f"  View at: https://huggingface.co/datasets/{repo_id}")
    print()
    print("Next steps for AutoTrain:")
    print(f"  1. Go to https://huggingface.co/spaces/autotrain-projects/autotrain-advanced")
    print(f"  2. Select 'LLM SFT' task")
    print(f"  3. Choose your dataset: {repo_id}")
    print(f"  4. Set text column: 'text'")
    print(f"  5. Choose base model: mistralai/Mistral-7B-Instruct-v0.2")
    print(f"  6. Start training!")

    return 0


if __name__ == "__main__":
    exit(main())
