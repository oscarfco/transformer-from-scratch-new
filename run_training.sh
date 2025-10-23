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
TOKENS_PROCESSED_TOTAL=40000000
BATCH_SIZE=32
LEARNING_RATE=0.01
DEVICE=mps
TRAINING_DSET_PATH="./data/tiny_stories_train.npy"

# optimizer args
WEIGHT_DECAY=0.01
BETA1=0.9
BETA2=0.999
EPS=1e-8

# save args
SAVE_EVERY=0.2

/Users/oscar.orahilly/miniconda3/bin/python training.py \
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
  --save-every "$SAVE_EVERY"
