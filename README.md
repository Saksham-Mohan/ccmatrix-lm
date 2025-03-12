# CCMatrix Language Model

This project implements a transformer-based language model trained on the CCMatrix dataset. It includes scripts for downloading data, tokenization, data processing, model training, and evaluation.

## Project Structure

```
ccmatrix-lm/
├── config/                # Configuration files
├── data/                  # Data processing scripts and processed data
│   ├── download.py        # Script to download CCMatrix data
│   └── dataset.py         # Script to create TensorFlow datasets
├── model/                 # Model architecture
│   └── transformer.py     # Transformer implementation
├── tokenizer/             # Tokenizer training and models
│   └── train_tokenizer.py # Script to train BPE tokenizer
├── training/              # Training scripts
│   └── train.py           # Main training script
└── utils/                 # Utility functions
    ├── checkpointing.py   # Checkpoint management
    └── logging.py         # Logging utilities
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ccmatrix-lm.git
cd ccmatrix-lm
```

2. Create and activate a conda environment:
```bash
conda create -n ccmatrix python=3.11
conda activate ccmatrix
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Pipeline

### 1. Download CCMatrix Data

```bash
python data/download.py --lang=en --output_dir=data/raw --max_sentences=10000
```

### 2. Train a BPE Tokenizer

```bash
python tokenizer/train_tokenizer.py \
    --input_file=data/processed/ccmatrix.en.processed.txt \
    --vocab_size=32000 \
    --output_dir=tokenizer/output
```

### 3. Process and Encode Data

```bash
python data/dataset.py \
    --tokenizer_path=tokenizer/output/tokenizer.json \
    --input_file=data/processed/ccmatrix.en.processed.txt \
    --output_file=data/processed/ccmatrix.en.encoded \
    --split
```

## Training

Train the model with the custom training loop:

```bash
python training/train.py \
    --lang=en \
    --tokenizer_path=tokenizer/output/tokenizer.json \
    --data_dir=data/processed \
    --checkpoint_dir=checkpoints/en \
    --tensorboard
```

Or use the built-in Keras training loop:

```bash
python training/train.py \
    --lang=en \
    --tokenizer_path=tokenizer/output/tokenizer.json \
    --data_dir=data/processed \
    --checkpoint_dir=checkpoints/en \
    --tensorboard \
    --use_fit
```

To train on a TPU (if available):

```bash
python training/train.py \
    --lang=en \
    --tokenizer_path=tokenizer/output/tokenizer.json \
    --data_dir=data/processed \
    --checkpoint_dir=checkpoints/en \
    --tensorboard \
    --tpu_name=YOUR_TPU_NAME
```

## Model Architecture

The model is based on the transformer architecture with:
- Multi-head self-attention
- Positional encoding
- Feed-forward neural networks
- Layer normalization

For debugging purposes, a simplified LSTM model is also available.

## Configuration

The model and training parameters can be customized in the configuration files:
- `config/model_config.yaml`: Model architecture parameters
- `config/training_config.yaml`: Training hyperparameters
