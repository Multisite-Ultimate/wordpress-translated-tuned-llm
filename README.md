# WordPress Translation LLM Fine-Tuning

Fine-tune Mistral 7B to translate WordPress content from English to Dutch using translation data from translate.wordpress.org.

## Overview

This project creates a specialized translation model for WordPress content by:

1. **Downloading** translation files (PO format) from the WordPress translation API
2. **Parsing** PO files to extract English→Dutch translation pairs
3. **Building** train/test datasets (80/20 split)
4. **Fine-tuning** Mistral 7B using LoRA (Low-Rank Adaptation)
5. **Evaluating** translation quality with BLEU, ChrF, and COMET metrics
6. **Serving** the model via REST API for integration

## Hardware Requirements

- **GPUs**: 2x NVIDIA GPUs with 10GB+ VRAM each
- **CUDA**: 6.1+ (tested with P102-100 mining GPUs)
- **RAM**: 32GB+ recommended
- **Storage**: 20GB+ for model weights and data

## Installation

```bash
# Clone and enter directory
cd wordpress-translated-tuned-llm

# Install dependencies
pip install -r requirements.txt

# For CUDA 6.1 GPUs (older cards), install compatible PyTorch:
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116
```

## Quick Start

```bash
# Run the full pipeline
./run.py download nl --limit 100    # Download Dutch translations
./run.py parse nl                    # Parse PO files to pairs
./run.py dataset nl                  # Build train/test datasets
./run.py train nl                    # Fine-tune the model
./run.py evaluate nl                 # Evaluate on test set
./run.py serve                       # Start REST API server
```

---

## Pipeline Commands

### 1. Download Translations

Download PO files from WordPress translation API:

```bash
# Download all Dutch translations (full dataset)
./run.py download nl

# Download with limits (for testing)
./run.py download nl --limit 100

# Download specific project types only
./run.py download nl --project-type wp-plugins
./run.py download nl --project-type wp-themes
./run.py download nl --project-type wp

# Adjust rate limiting
./run.py download nl --rate-limit 60
```

**Output**: `data/raw/nl/*.po` files

### 2. Parse Translation Pairs

Extract translation pairs from PO files:

```bash
# Parse all downloaded PO files
./run.py parse nl

# Custom input/output directories
./run.py parse nl --input ./data/raw --output ./data/processed

# Show sample translation pairs
./run.py parse nl --show-samples 10
```

**Filtering applied**:
- Skips fuzzy/incomplete translations
- Skips empty translations
- Skips identical source/target
- Preserves placeholders (%s, %d, {name})

**Output**: `data/processed/nl/pairs.jsonl`

### 3. Build Dataset

Create train/test split for fine-tuning:

```bash
# Build with default 80/20 split
./run.py dataset nl

# Custom test split ratio
./run.py dataset nl --test-size 0.1

# Use different base model for tokenization
./run.py dataset nl --model mistralai/Mistral-7B-v0.1

# Custom max sequence length
./run.py dataset nl --max-length 256
```

**Output**: `data/datasets/nl/` (HuggingFace dataset format)

### 4. Train Model

Fine-tune Mistral 7B with LoRA:

```bash
# Basic training (auto-detects GPU capabilities)
./run.py train nl

# Custom epochs and batch size
./run.py train nl --epochs 3 --batch-size 1

# Adjust learning rate and LoRA rank
./run.py train nl --lr 1e-4 --lora-r 32

# Use custom config file
./run.py train nl --config configs/training/qlora_mistral.yaml

# Resume from checkpoint
./run.py train nl --resume models/checkpoints/nl/checkpoint-1000
```

**Training configuration**:
- Base model: `mistralai/Mistral-7B-Instruct-v0.2`
- LoRA rank: 64, alpha: 16
- Learning rate: 2e-4
- Gradient accumulation: 16 steps
- FP16 precision (FP32 gradients)

**Output**: `models/adapters/nl/` (LoRA weights)

### 5. Evaluate Model

Test translation quality on held-out test set:

```bash
# Evaluate on test set
./run.py evaluate nl

# Specify adapter path
./run.py evaluate nl --model models/adapters/nl

# Limit number of samples
./run.py evaluate nl --max-samples 500

# Skip COMET metric (faster)
./run.py evaluate nl --no-comet

# Test single translation
./run.py evaluate translate "Hello world" --model models/adapters/nl
```

**Metrics computed**:
- **BLEU**: N-gram precision (0-100)
- **ChrF**: Character-level F-score (0-100)
- **COMET**: Neural metric correlating with human judgment (-1 to 1)

**Output**: `logs/evaluation/nl/` (JSON reports with sample translations)

### 6. Serve Model

Start REST API for translation:

```bash
# Start server on default port
./run.py serve

# Custom port and host
./run.py serve --port 8080 --host 0.0.0.0

# Specify adapter
./run.py serve --model models/adapters/nl
```

**API endpoints**:
```bash
# Translate single text
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Add to cart", "source_lang": "en", "target_lang": "nl"}'

# Batch translation
curl -X POST http://localhost:8000/translate/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Add to cart", "Checkout"], "source_lang": "en", "target_lang": "nl"}'
```

---

## Project Structure

```
wordpress-translated-tuned-llm/
├── run.py                      # Main CLI entry point
├── requirements.txt            # Python dependencies
├── configs/
│   └── training/
│       └── qlora_mistral.yaml  # Training configuration
├── src/wp_translation/
│   ├── downloader/             # WordPress API client & PO fetcher
│   │   ├── client.py           # Rate-limited HTTP client
│   │   ├── api_client.py       # WordPress.org API wrapper
│   │   └── fetcher.py          # PO file downloader
│   ├── parser/                 # PO file parsing
│   │   ├── po_parser.py        # Extract msgid/msgstr pairs
│   │   ├── cleaner.py          # Text normalization
│   │   └── pair_extractor.py   # Quality filtering
│   ├── dataset/                # Dataset building
│   │   ├── formatter.py        # Mistral prompt formatting
│   │   ├── splitter.py         # Train/test split
│   │   └── builder.py          # HuggingFace dataset creation
│   ├── training/               # Fine-tuning pipeline
│   │   ├── config.py           # Training configuration
│   │   ├── model_loader.py     # Model loading with quantization
│   │   ├── lora_config.py      # LoRA adapter configuration
│   │   └── trainer.py          # Training loop
│   ├── evaluation/             # Quality metrics
│   │   ├── metrics.py          # BLEU, ChrF, COMET
│   │   ├── evaluator.py        # Test set evaluation
│   │   └── report.py           # Report generation
│   ├── inference/              # Translation serving
│   │   ├── translator.py       # Translation interface
│   │   └── server.py           # FastAPI server
│   └── utils/                  # Utilities
│       ├── config.py           # Configuration loading
│       ├── logging.py          # Logging setup
│       └── gpu.py              # GPU utilities
├── cli/commands/               # CLI command implementations
├── data/
│   ├── raw/nl/                 # Downloaded PO files
│   ├── processed/nl/           # Parsed translation pairs
│   └── datasets/nl/            # Train/test datasets
├── models/
│   ├── checkpoints/            # Training checkpoints
│   ├── adapters/               # LoRA adapter weights
│   └── final/                  # Merged/exported models
└── logs/                       # Training and evaluation logs
```

---

## Technical Details

### Prompt Format

The model uses the Mistral instruction format:

```
<s>[INST] Translate the following WordPress text from English to Dutch.
Preserve any placeholders like %s, %d, or {name}.

Add to cart [/INST]Toevoegen aan winkelwagen</s>
```

### GPU Memory Usage

With 2x 10GB GPUs and FP16 precision:

| Component | Memory |
|-----------|--------|
| Base model (FP16) | ~14 GB (split across GPUs) |
| LoRA adapters | ~0.3 GB |
| Optimizer states | ~1.5 GB |
| Gradients + activations | ~3.5 GB |
| **Total** | ~19 GB |

For GPUs with compute capability < 7.5 (like P102-100), 4-bit quantization is not available. The model automatically falls back to FP16 and uses model parallelism across multiple GPUs.

### Data Statistics

Example dataset for Dutch (nl):
- **Training examples**: 10,890
- **Test examples**: 2,723
- **Average source length**: 40 characters
- **Average target length**: 46 characters
- **Project distribution**: plugins (23%), themes (8%), core (69%)

---

## Troubleshooting

### CUDA out of memory
- Reduce batch size: `./run.py train nl --batch-size 1`
- Increase gradient accumulation: edit `configs/training/qlora_mistral.yaml`
- Reduce max sequence length: `./run.py dataset nl --max-length 256`

### Slow download
- The WordPress API has rate limits; downloads may take time for large datasets
- Use `--limit` to test with fewer projects first

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- For CUDA 6.1: use PyTorch 1.12.1 as shown in installation

### bitsandbytes GPU unavailable
- CUDA 6.1 GPUs don't support efficient 8-bit operations
- The trainer automatically uses FP16 instead of quantization

---

## License

This project is for educational and research purposes. WordPress translations are contributed by volunteers under GPL-compatible licenses.
