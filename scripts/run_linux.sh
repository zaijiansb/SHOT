#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
DATASETS_ROOT="../Datasets"
SOURCE_CHECKPOINT="checkpoints/source.pt"
SOURCE_EPOCHS=20
TARGET_EPOCHS=15
BATCH_SIZE=32

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --datasets-root)
      DATASETS_ROOT="$2"
      shift 2
      ;;
    --source-checkpoint)
      SOURCE_CHECKPOINT="$2"
      shift 2
      ;;
    --source-epochs)
      SOURCE_EPOCHS="$2"
      shift 2
      ;;
    --target-epochs)
      TARGET_EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cd "$(dirname "$0")/.."

echo "==> Train source domain: AWGN.dat"
"$PYTHON_BIN" scripts/train_source.py \
  --data-root "$DATASETS_ROOT/AWGN.dat" \
  --output "$SOURCE_CHECKPOINT" \
  --epochs "$SOURCE_EPOCHS" \
  --batch-size "$BATCH_SIZE"

targets=(
  "Rayleigh1.dat"
  "Rayleigh2.dat"
  "Rayleigh3.dat"
  "Rician1.dat"
  "Rician3.dat"
)

for target in "${targets[@]}"; do
  echo "==> Adapt target domain: $target"
  "$PYTHON_BIN" scripts/adapt_target.py \
    --data-root "$DATASETS_ROOT/$target" \
    --source-checkpoint "$SOURCE_CHECKPOINT" \
    --target-split train \
    --eval-split val \
    --epochs "$TARGET_EPOCHS" \
    --batch-size "$BATCH_SIZE"
done

echo "==> Done. Results are saved in results/"
