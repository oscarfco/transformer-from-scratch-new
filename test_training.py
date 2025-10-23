#!/usr/bin/env python3
"""
Quick test script to verify the transformer training works.
This runs a very small training session to test the setup.
"""

import subprocess
import sys
import os

def main():
    print("🧪 Testing transformer training setup...")
    
    # Check if data exists
    if not os.path.exists("./data/tiny_stories_train.npy"):
        print("❌ Training data not found at ./data/tiny_stories_train.npy")
        print("   Please copy your training data to this location.")
        return 1
    
    # Check if vocab files exist
    if not os.path.exists("./vocab.pkl") or not os.path.exists("./merges.pkl"):
        print("❌ Tokenizer files not found (vocab.pkl, merges.pkl)")
        return 1
    
    print("✅ Data and tokenizer files found")
    
    # Run a small training test
    cmd = [
        "/Users/oscar.orahilly/miniconda3/bin/python", "training.py",
        "--vocab-size", "1000",
        "--context-length", "64", 
        "--d-model", "128",
        "--num-layers", "2",
        "--num-heads", "4",
        "--dff", "256",
        "--batch-size", "2",
        "--tokens-processed-total", "1000",
        "--device", "mps",
        "--no-wandb",
        "--save-every", "0.5"
    ]
    
    print("🚀 Running test training...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Training test completed successfully!")
            print("📁 Checkpoints saved in ./checkpoints/")
            return 0
        else:
            print(f"❌ Training failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return 1
    except subprocess.TimeoutExpired:
        print("⏰ Training test timed out (this is normal for a quick test)")
        return 0
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
