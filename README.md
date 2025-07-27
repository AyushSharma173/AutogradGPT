# AutogradGPT

A full Transformer-based language model built from scratch in PyTorch — including custom tokenizer (BPE), RoPE attention, RMSNorm, SwiGLU, AdamW, and a training pipeline with text generation, checkpointing, and W&B logging.

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AyushSharma173/AutogradGPT.git
cd AutogradGPT
```

### 2. Set Up Environment
Create a virtual environment and install dependencies using `uv`:

```bash
# Create virtual environment
uv venv

# Activate the environment (Linux/Mac)
source .venv/bin/activate
# On Windows: .venv\Scripts\activate

# Install locked dependencies
uv pip install --require-locked
```

Alternatively, if you prefer using other package managers:
```bash
pip install -r requirements.txt  # if you have a requirements.txt
```

### 3. Download Training Data
We provide a convenient script to download all required datasets:

```bash
./download_data.sh
```

This script will download:
- **TinyStories datasets** (from Hugging Face):
  - `TinyStoriesV2-GPT4-train.txt` (~2.1GB) - Training data
  - `TinyStoriesV2-GPT4-valid.txt` (~21MB) - Validation data

- **OpenWebText sample datasets** (from Stanford CS336):
  - `owt_train.txt` - Training data (compressed)
  - `owt_valid.txt` - Validation data (compressed)

All files will be downloaded to the `data/` directory.

### Manual Download (Alternative)
If you prefer to download manually:

```bash
mkdir -p data
cd data

# TinyStories datasets
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText sample datasets  
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Usage

### Tokenization
Run the tokenizer to process your datasets:
```bash
python tokenizer.py
```


## Project Structure
```
AutogradGPT/
├── README.md              # This file
├── download_data.sh       # Data download script
├── tokenizer.py          # BPE tokenizer implementation
├── modeling.py           # Transformer model architecture
├── bpe.py               # BPE training utilities
├── utils.py             # Helper functions
├── vocab.json           # Trained vocabulary
├── merges.json          # BPE merge rules
└── data/                # Training data (after running download script)
    ├── TinyStoriesV2-GPT4-train.txt
    ├── TinyStoriesV2-GPT4-valid.txt
    ├── owt_train.txt
    └── owt_valid.txt
```

## Features
- Custom BPE tokenizer from scratch
- Full Transformer architecture with modern improvements:
  - RoPE (Rotary Position Embedding)
  - RMSNorm normalization
  - SwiGLU activation function
- AdamW optimizer
- Training pipeline with checkpointing
- Weights & Biases integration
- Text generation capabilities 