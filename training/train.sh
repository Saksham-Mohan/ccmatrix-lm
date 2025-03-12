#!/bin/bash
# Shell script to train a transformer language model on CCMatrix data

set -e  # Exit on error

LANG="en"
MAX_TOKENS=1000000
TPU_NAME=""
TENSORBOARD=0
WANDB=0
CHECKPOINT_PATH=""

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
    --tpu_name=*)
      TPU_NAME="${arg#*=}"
      shift
      ;;
    --tensorboard)
      TENSORBOARD=1
      shift
      ;;
    --wandb)
      WANDB=1
      shift
      ;;
    --checkpoint_path=*)
      CHECKPOINT_PATH="${arg#*=}"
      shift
      ;;
    *)
      # Unknown option
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

DATA_DIR="data/processed"
TOKENIZER_DIR="tokenizer/output"
CHECKPOINT_DIR="checkpoints/${LANG}"
LOG_DIR="logs"

# Create directories if they don't exist
mkdir -p "${DATA_DIR}"
mkdir -p "${TOKENIZER_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${LOG_DIR}"

echo "Starting training pipeline for language: ${LANG}"
echo "Maximum tokens: ${MAX_TOKENS}"

# Step 1: Download and preprocess data
echo "Step 1: Downloading and preprocessing CCMatrix data..."
python data/download.py --lang="${LANG}" --output_dir="data/raw" --processed_dir="${DATA_DIR}" --max_sentences="${MAX_TOKENS}"

# Step 2: Train tokenizer
echo "Step 2: Training BPE tokenizer..."
TOKENIZER_PATH="${TOKENIZER_DIR}/tokenizer_${LANG}.json"

if [ ! -f "${TOKENIZER_PATH}" ]; then
  python tokenizer/train_tokenizer.py \
    --input_file="${DATA_DIR}/ccmatrix.${LANG}.processed.txt" \
    --vocab_size=32000 \
    --output_dir="${TOKENIZER_DIR}" \
    --max_tokens="${MAX_TOKENS}"
  
  # Rename the tokenizer file
  mv "${TOKENIZER_DIR}/tokenizer.json" "${TOKENIZER_PATH}"
else
  echo "Tokenizer already exists at ${TOKENIZER_PATH}. Skipping training."
fi

# Step 3: Train the model
echo "Step 3: Training transformer model..."

# Prepare training command
TRAIN_CMD="python training/train.py \
  --lang=\"${LANG}\" \
  --tokenizer_path=\"${TOKENIZER_PATH}\" \
  --data_dir=\"${DATA_DIR}\" \
  --checkpoint_dir=\"${CHECKPOINT_DIR}\""

# Add TPU argument if provided
if [ ! -z "${TPU_NAME}" ]; then
  TRAIN_CMD="${TRAIN_CMD} --tpu_name=\"${TPU_NAME}\""
fi

# Add TensorBoard flag if enabled
if [ "${TENSORBOARD}" -eq 1 ]; then
  TRAIN_CMD="${TRAIN_CMD} --tensorboard"
fi

# Add Weights & Biases flag if enabled
if [ "${WANDB}" -eq 1 ]; then
  TRAIN_CMD="${TRAIN_CMD} --wandb"
fi

# Add checkpoint path if provided
if [ ! -z "${CHECKPOINT_PATH}" ]; then
  TRAIN_CMD="${TRAIN_CMD} --checkpoint_path=\"${CHECKPOINT_PATH}\""
fi

# Add max tokens argument
TRAIN_CMD="${TRAIN_CMD} --max_tokens=\"${MAX_TOKENS}\""

# Execute the training command
echo "Executing: ${TRAIN_CMD}"
eval "${TRAIN_CMD}"

echo "Training pipeline completed!"