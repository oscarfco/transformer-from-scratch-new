#!/bin/bash
# Learning Rate Sweep Script
# Runs training with learning rates from 0.05 to 0.1 in increments of 0.01
# Min LR is 50% of Max LR for each experiment

set -euo pipefail

# Base configuration (same as original)
VOCAB_SIZE=10000
CONTEXT_LENGTH=256
D_MODEL=512
D_FF=1344
ROPE_THETA=10000
NUM_LAYERS=4
NUM_HEADS=16
TOKENS_PROCESSED_TOTAL=70000000
BATCH_SIZE=128
DEVICE=cuda
TRAINING_DSET_PATH="./data/tiny_stories_train.npy"
TOKENIZER_PATH="./data/tiny_stories_train.pkl"
EVAL_PROMPT="Once upon a time"

# optimizer args
WEIGHT_DECAY=0.001
BETA1=0.9
BETA2=0.999
EPS=1e-8

# scheduler args - MIN_LR will be calculated as 50% of MAX_LR
WARMUP_ITERS=$(echo "scale=0; ($TOKENS_PROCESSED_TOTAL / ($BATCH_SIZE * $CONTEXT_LENGTH)) * 0.01 / 1" | bc)
COSINE_CYCLE_ITERS=$(echo "scale=0; $TOKENS_PROCESSED_TOTAL / ($BATCH_SIZE * $CONTEXT_LENGTH)" | bc)

# save args
SAVE_EVERY=0.2
LOG_EVERY=0.1
NUM_EVAL_GENERATIONS=1

# Learning rate sweep parameters
LR_START=0.3
LR_END=0.1
LR_INCREMENT=-0.05

# Create results directory
mkdir -p lr_sweep_results

echo "Starting Learning Rate Sweep"
echo "Learning rates to test: $(seq $LR_START $LR_INCREMENT $LR_END | tr '\n' ' ')"
echo "Min LR will be 90% of Max LR for each experiment (10% reduction)"
echo "Results will be saved in: lr_sweep_results/"
echo "=========================================="

# Loop through learning rates
for lr in $(seq $LR_START $LR_INCREMENT $LR_END); do
    # Calculate min_lr as 90% of max_lr (10% reduction)
    min_lr=$(echo "scale=3; $lr * 0.9" | bc)
    
    echo ""
    echo "Training with Learning Rate: $lr"
    echo "Max LR: $lr, Min LR: $min_lr"
    echo "----------------------------------------"
    
    # Create a unique run name for this learning rate
    RUN_NAME="lr_${lr}_minlr_${min_lr}"
    
    # Run training with current learning rate
    python training.py \
      --vocab-size "$VOCAB_SIZE" \
      --context-length "$CONTEXT_LENGTH" \
      --d-model "$D_MODEL" \
      --dff "$D_FF" \
      --rope-theta "$ROPE_THETA" \
      --num-layers "$NUM_LAYERS" \
      --num-heads "$NUM_HEADS" \
      --tokens-processed-total "$TOKENS_PROCESSED_TOTAL" \
      --batch-size "$BATCH_SIZE" \
      --learning-rate "$lr" \
      --weight-decay "$WEIGHT_DECAY" \
      --beta1 "$BETA1" \
      --beta2 "$BETA2" \
      --eps "$EPS" \
      --device "$DEVICE" \
      --training-dset-path "$TRAINING_DSET_PATH" \
      --save-every "$SAVE_EVERY" \
      --log-every "$LOG_EVERY" \
      --tokenizer-path "$TOKENIZER_PATH" \
      --eval-prompt "$EVAL_PROMPT" \
      --num-eval-generations "$NUM_EVAL_GENERATIONS" \
      --max-lr "$lr" \
      --min-lr "$min_lr" \
      --warmup-iters "$WARMUP_ITERS" \
      --cosine-cycle-iters "$COSINE_CYCLE_ITERS"
    
    # Move checkpoints to results directory with LR-specific naming
    if [ -d "checkpoints" ]; then
        mkdir -p "lr_sweep_results/${RUN_NAME}"
        mv checkpoints/* "lr_sweep_results/${RUN_NAME}/" 2>/dev/null || true
        echo "Checkpoints saved to: lr_sweep_results/${RUN_NAME}/"
    fi
    
    echo "Completed training with LR: $lr"
done

echo ""
echo "=========================================="
echo "Learning Rate Sweep Complete!"
echo "All results saved in: lr_sweep_results/"
echo "Learning rates tested: $(seq $LR_START $LR_INCREMENT $LR_END | tr '\n' ' ')"
echo "Min LR was 90% of Max LR for each experiment (10% reduction)"

