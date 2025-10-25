import torch 
import numpy as np
from training_modules import cross_entropy
from tqdm import tqdm 

def get_eval_batch(dataset, batch_size, i, context_length, device):
    start_indices = torch.tensor([i+(context_length*j) for j in range(batch_size)])

    input_sequences = torch.empty(batch_size, context_length, dtype=torch.int64)
    token_targets = torch.empty(batch_size, context_length, dtype=torch.int64)

    for b in range(batch_size):
        start_idx = start_indices[b]
        end_idx = start_idx + context_length
        input_sequences[b, :] = torch.tensor(dataset[start_idx:end_idx])
        token_targets[b, :] = torch.tensor(dataset[start_idx+1: end_idx+1])

    return input_sequences.to(device), token_targets.to(device)


def load_data(path):
    # Loads the data in this lazy way
    return np.load(path, mmap_mode='r')


# 1. Run validation oin the eval set 
def run_eval(model, tokenizer, batch_size, context_length, device, eval_path="data/tiny_stories_valid.npy"):
    eval_dset = load_data(eval_path)

    n = len(eval_dset)
    eval_loss = []

    for i in tqdm(range(0, n, batch_size*context_length)):
        if n - i < + batch_size*context_length:
            break
        
        inputs, targets = get_eval_batch(eval_dset, batch_size, i, context_length, device)

        with torch.no_grad():
            outputs = model(inputs)

        loss = cross_entropy(outputs, targets)
        
        eval_loss.append(loss.item())
    
    return np.mean(eval_loss)
    

if __name__ == "__main__":
    data = load_data("data/tiny_stories_valid.npy")
    breakpoint()






# 2. Log some completions during training