#!/bin/bash
# Run the complete pretraining pipeline

set -e  # Exit on error

if command -v conda &> /dev/null; then
    echo "Activating ccmatrix conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate ccmatrix
else
    echo "Conda not found, continuing with current environment..."
fi

LANG="en"
MAX_TOKENS=10000  # Reduced for testing
USE_TPU=0
TPU_NAME=""
USE_TENSORBOARD=1
USE_WANDB=0

for arg in "$@"; do
  case $arg in
    --lang=*)
      LANG="${arg#*=}"
      shift
      ;;
    --max_tokens=*)
      MAX_TOKENS="${arg#*=}"
      shift
      ;;
    --use_tpu)
      USE_TPU=1
      shift
      ;;
    --tpu_name=*)
      TPU_NAME="${arg#*=}"
      USE_TPU=1
      shift
      ;;
    --use_wandb)
      USE_WANDB=1
      shift
      ;;
    --no_tensorboard)
      USE_TENSORBOARD=0
      shift
      ;;
    *)
      # Unknown option
      echo "Unknown option: $arg"
      echo "Usage: $0 [--lang=<language_code>] [--max_tokens=<number>] [--use_tpu] [--tpu_name=<name>] [--use_wandb] [--no_tensorboard]"
      exit 1
      ;;
  esac
done

mkdir -p data/raw
mkdir -p data/processed
mkdir -p tokenizer/output
mkdir -p checkpoints
mkdir -p logs

echo "Starting pipeline for language: $LANG with max_tokens: $MAX_TOKENS"

echo "Step 1: Downloading and preprocessing data..."
python data/download.py --lang="$LANG" --output_dir="data/raw" --processed_dir="data/processed" --max_sentences="$MAX_TOKENS"

echo "Step 2: Training tokenizer..."
python tokenizer/train_tokenizer.py --input_file="data/processed/ccmatrix.$LANG.processed.txt" --vocab_size=32000 --output_dir="tokenizer/output" --max_tokens="$MAX_TOKENS"

# Step 3: Create datasets
echo "Step 3: Creating TensorFlow datasets..."
python data/dataset.py --tokenizer_path="tokenizer/output/tokenizer.json" --input_file="data/processed/ccmatrix.$LANG.processed.txt" --output_file="data/processed/ccmatrix.$LANG.encoded" --sequence_length=512 --split

# Step 4: Training
echo "Step 4: Training model..."

TRAIN_CMD="python training/train.py \
  --lang=\"$LANG\" \
  --tokenizer_path=\"tokenizer/output/tokenizer.json\" \
  --data_dir=\"data/processed\" \
  --train_encoded_file=\"data/processed/ccmatrix.$LANG.encoded.train\" \
  --val_encoded_file=\"data/processed/ccmatrix.$LANG.encoded.val\" \
  --checkpoint_dir=\"checkpoints/$LANG\" \
  --log_dir=\"logs\" \
  --max_tokens=\"$MAX_TOKENS\""

if [ "$USE_TPU" -eq 1 ]; then
  TRAIN_CMD="$TRAIN_CMD --tpu_name=\"$TPU_NAME\""
fi

# Add TensorBoard flag if enabled
if [ "$USE_TENSORBOARD" -eq 1 ]; then
  TRAIN_CMD="$TRAIN_CMD --tensorboard"
fi

# Add Weights & Biases flag if enabled
if [ "$USE_WANDB" -eq 1 ]; then
  TRAIN_CMD="$TRAIN_CMD --wandb"
fi

# Execute the training command
echo "Executing: $TRAIN_CMD"
eval "$TRAIN_CMD"

echo "Pipeline completed successfully!"