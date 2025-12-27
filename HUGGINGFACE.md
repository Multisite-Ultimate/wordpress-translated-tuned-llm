# Hugging Face Integration Guide

This guide explains how to run the entire WordPress Translation LLM project on Hugging Face infrastructure.

## Overview

| Component | Local | Hugging Face |
|-----------|-------|--------------|
| Dataset storage | `data/datasets/` | HF Datasets Hub |
| Training | Local GPU / Colab | AutoTrain or Spaces |
| Model hosting | `models/adapters/` | HF Model Hub |
| Inference | Local FastAPI | HF Inference Endpoints |

## Quick Start

### 1. Login to Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login
```

### 2. Upload Your Dataset

```bash
# First, build your dataset locally
./run.py download nl --limit 100
./run.py parse nl
./run.py dataset build nl

# Then push to Hugging Face
./run.py dataset push nl
```

This creates: `https://huggingface.co/datasets/YOUR_USERNAME/wordpress-translations-nl`

### 3. Train with AutoTrain

1. Go to [AutoTrain Advanced](https://huggingface.co/spaces/autotrain-projects/autotrain-advanced)
2. Select **LLM SFT** task
3. Configure:
   - **Model**: `mistralai/Mistral-7B-Instruct-v0.2`
   - **Dataset**: `YOUR_USERNAME/wordpress-translations-nl`
   - **Text Column**: `text`
   - **Train Split**: `train`
   - **Validation Split**: `test`

4. Training Settings (recommended):
   ```
   Epochs: 1
   Batch Size: 2
   Gradient Accumulation: 8
   Learning Rate: 0.0002
   Use PEFT/LoRA: Yes
   LoRA r: 32
   LoRA alpha: 16
   Quantization: int4
   ```

5. Click **Start Training**

Estimated cost: **$5-15** for 1 epoch on A10G GPU.

### 4. Push Trained Model (if training locally)

```bash
# After local training completes
./run.py train push nl
```

This creates: `https://huggingface.co/YOUR_USERNAME/mistral-7b-wordpress-nl`

## Detailed Workflows

### Option A: Full AutoTrain (No-Code)

Best for: Users who want simplicity without local GPU.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Local      │     │  HF Hub     │     │  AutoTrain  │
│  Dataset    │ ──► │  Dataset    │ ──► │  Training   │
│  Build      │     │  Storage    │     │  (A10G/A100)│
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  HF Hub     │
                                        │  Model      │
                                        └─────────────┘
```

Steps:
1. Build dataset locally (no GPU needed)
2. Push dataset to HF Hub
3. Train with AutoTrain UI
4. Model auto-pushed to your HF account

### Option B: Spaces with GPU

Best for: Users who want more control over training.

1. Create a new Space with GPU (A10G recommended)
2. Upload your training code
3. Run training in the Space
4. Push model to Hub

Example Space setup:
```python
# app.py in your HF Space
import gradio as gr
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

def train(dataset_id, epochs, lr):
    dataset = load_dataset(dataset_id)
    # ... training code ...
    return "Training complete!"

demo = gr.Interface(
    fn=train,
    inputs=[
        gr.Textbox(label="Dataset ID"),
        gr.Slider(1, 5, value=1, label="Epochs"),
        gr.Number(value=0.0002, label="Learning Rate"),
    ],
    outputs="text"
)
demo.launch()
```

### Option C: Hybrid (Local + Hub)

Best for: Users with local GPU who want Hub hosting.

1. Train locally: `./run.py train run nl`
2. Push model: `./run.py train push nl`
3. Use via HF Inference Endpoints

## CLI Commands Reference

### Dataset Commands

```bash
# Build dataset from downloaded PO files
./run.py dataset build nl

# View dataset info
./run.py dataset info nl

# Push dataset to Hugging Face Hub
./run.py dataset push nl

# Push to organization
./run.py dataset push nl --org my-organization

# Push as private dataset
./run.py dataset push nl --private
```

### Model Commands

```bash
# Push trained model to Hub
./run.py train push nl

# Push merged model (not just adapters)
./run.py train push nl --merged

# Push to organization
./run.py train push nl --org my-organization

# Specify custom adapter path
./run.py train push nl --adapter ./my-custom-adapter
```

## Dataset Format

The dataset on HF Hub contains these columns:

| Column | Description |
|--------|-------------|
| `text` | Full training example with Mistral instruction format |
| `prompt` | Inference prompt (without target) |
| `source` | Original English text |
| `target` | Translation |
| `project_type` | wp-plugins, wp-themes, etc. |
| `project_name` | Original project name |

Example:
```python
from datasets import load_dataset

ds = load_dataset("YOUR_USERNAME/wordpress-translations-nl")
print(ds["train"][0])

# Output:
# {
#   "text": "<s>[INST] Translate... [/INST]Translation here</s>",
#   "source": "Add to cart",
#   "target": "Toevoegen aan winkelwagen",
#   ...
# }
```

## Model Usage

### From HF Hub (Adapter)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    load_in_4bit=True
)

# Load adapter
model = PeftModel.from_pretrained(
    base_model,
    "YOUR_USERNAME/mistral-7b-wordpress-nl"
)

tokenizer = AutoTokenizer.from_pretrained(
    "YOUR_USERNAME/mistral-7b-wordpress-nl"
)

# Translate
prompt = """<s>[INST] Translate the following WordPress text from English to Dutch.
Preserve any placeholders like %s, %d, or {name}.

Add to cart [/INST]"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Using Inference Endpoints

1. Go to your model on HF Hub
2. Click "Deploy" → "Inference Endpoints"
3. Select GPU and region
4. Deploy

Then call via API:
```python
import requests

API_URL = "https://your-endpoint.huggingface.cloud"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

response = requests.post(API_URL, headers=headers, json={
    "inputs": "Translate: Add to cart → Dutch:"
})
print(response.json())
```

## Cost Estimates

### AutoTrain

| GPU | $/hour | 1 epoch (~200k examples) | 3 epochs |
|-----|--------|--------------------------|----------|
| A10G (24GB) | $1.05 | ~$5-7 | ~$15-20 |
| A100-40GB | $4.13 | ~$8-12 | ~$25-35 |

### Inference Endpoints

| Size | GPU | $/hour |
|------|-----|--------|
| Small | T4 | $0.60 |
| Medium | A10G | $1.30 |
| Large | A100 | $6.50 |

### Storage

- Datasets: Free (public), $9/mo Pro (private)
- Models: Free (public), $9/mo Pro (private)

## Troubleshooting

### "Token not found"

Run `huggingface-cli login` and paste your token from https://huggingface.co/settings/tokens

### "Dataset not found"

Make sure you've pushed the dataset first:
```bash
./run.py dataset push nl
```

### AutoTrain training fails

1. Check your dataset format - `text` column should contain the full training example
2. Try reducing batch size or LoRA rank
3. Check the AutoTrain logs for specific errors

### Out of memory on Spaces

1. Use 4-bit quantization (`load_in_4bit=True`)
2. Reduce batch size to 1
3. Upgrade to larger GPU (A10G → A100)

## Resources

- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub)
- [AutoTrain Documentation](https://huggingface.co/docs/autotrain)
- [PEFT/LoRA Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
