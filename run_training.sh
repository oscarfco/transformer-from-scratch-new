#!/bin/bash
# Usage: ./run_training.sh
set -euo pipefail

VOCAB_SIZE=10000
CONTEXT_LENGTH=256
D_MODEL=512
D_FF=1344
ROPE_THETA=10000
NUM_LAYERS=4
NUM_HEADS=16
TOKENS_PROCESSED_TOTAL=95000000
BATCH_SIZE=128
LEARNING_RATE=0.01
DEVICE=cuda
TRAINING_DSET_PATH="./data/tiny_stories_train.npy"
TOKENIZER_PATH="./data/tiny_stories_train.pkl"
EVAL_PROMPT="Once upon a time"

# optimizer args
WEIGHT_DECAY=0.01
BETA1=0.9
BETA2=0.999
EPS=1e-8

# scheduler args
MAX_LR=0.001
MIN_LR=0.0001
WARMUP_ITERS=$(echo "scale=0; ($TOKENS_PROCESSED_TOTAL / ($BATCH_SIZE * $CONTEXT_LENGTH)) * 0.01 / 1" | bc)
COSINE_CYCLE_ITERS=$(echo "scale=0; $TOKENS_PROCESSED_TOTAL / ($BATCH_SIZE * $CONTEXT_LENGTH)" | bc)

# save args
SAVE_EVERY=0.2
LOG_EVERY=0.1
NUM_EVAL_GENERATIONS=1

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
  --learning-rate "$LEARNING_RATE" \
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
  --max-lr "$MAX_LR" \
  --min-lr "$MIN_LR" \
  --warmup-iters "$WARMUP_ITERS" \
  --cosine-cycle-iters "$COSINE_CYCLE_ITERS"
