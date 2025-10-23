# Transformer from Scratch

A standalone implementation of a transformer language model trained from scratch, extracted from CS336 Assignment 1.

## Features

- **Complete Transformer Architecture**: Implements multi-head self-attention with RoPE (Rotary Positional Embedding)
- **Custom Optimizer**: AdamW optimizer implementation
- **BPE Tokenizer**: Byte-pair encoding tokenizer with special tokens
- **Training Infrastructure**: Full training loop with checkpointing and logging
- **Wandb Integration**: Experiment tracking and visualization

## Architecture

- **Model**: Transformer Language Model with RMSNorm and SwiGLU activation
- **Attention**: Multi-head self-attention with RoPE positional encoding
- **Optimizer**: Custom AdamW implementation
- **Tokenizer**: BPE tokenizer with special token support

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   - Place your training data in `./data/tiny_stories_train.npy`
   - The data should be a numpy array of tokenized text

3. **Run Training**:
   ```bash
   ./run_training.sh
   ```

## Configuration

The training script supports various hyperparameters:

- `VOCAB_SIZE`: Vocabulary size (default: 10000)
- `CONTEXT_LENGTH`: Sequence length (default: 256)
- `D_MODEL`: Model dimension (default: 512)
- `NUM_LAYERS`: Number of transformer layers (default: 4)
- `NUM_HEADS`: Number of attention heads (default: 16)
- `BATCH_SIZE`: Training batch size (default: 32)
- `LEARNING_RATE`: Learning rate (default: 0.01)
- `DEVICE`: Device to use (default: mps for Apple Silicon)

## Files

- `training.py`: Main training script
- `transformer_modules.py`: Transformer architecture implementation
- `training_modules.py`: Training utilities (optimizer, loss functions, etc.)
- `tokenizer.py`: BPE tokenizer implementation
- `run_training.sh`: Training entry point script
- `vocab.pkl`: Vocabulary file for tokenizer
- `merges.pkl`: BPE merges file for tokenizer

## Training

The training script will:
1. Load the training data
2. Initialize the transformer model
3. Set up the AdamW optimizer
4. Run the training loop with progress tracking
5. Save checkpoints periodically
6. Log metrics to Wandb (if enabled)

## Checkpoints

Model checkpoints are saved in the `checkpoints/` directory:
- `model_step_{step}.pt`: Periodic checkpoints
- `model_step_final.pt`: Final model checkpoint

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA/MPS support for GPU training
- Wandb account for experiment tracking (optional)
# transformer-from-scratch-new
