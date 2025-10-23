import torch 
import torch.nn as nn
import numpy as np
import math
from einops import einsum, rearrange, reduce
from collections.abc import Callable, Iterable
from typing import Optional
from tqdm import tqdm 
import random

def cross_entropy(logits, targets):
    # logits = Float[Tensor, " batch_size vocab_size"]
    # targets = Int[Tensor, " batch_size"]

    # torch.gather(input, dim, index, *, sparse_grad=False, out=None)
    # input and index must be the same shape

    targets = rearrange(targets, '... B -> ... B 1')
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    adjusted_logits = logits - max_logits
    numerator = torch.gather(adjusted_logits, dim=-1, index=targets).squeeze(1) # (B)

    denom = reduce(torch.exp(adjusted_logits), '... B V -> ... B 1', 'sum')
    log_denom = torch.log(denom)

    return reduce(numerator-log_denom, '... B -> ', 'mean') * -1


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "b1": betas[0], "b2": betas[1], "e": eps, "l": weight_decay}

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate
            b1 = group["b1"]
            b2 = group["b2"]
            e = group["e"]
            l = group["l"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.

                first_moment = state.get("first_moment", torch.zeros_like(p))
                second_moment = state.get("second_moment", torch.zeros_like(p))

                state["first_moment"] = b1*first_moment + (1-b1)*grad
                state["second_moment"] = b2*second_moment + (1-b2)*torch.pow(grad, 2)

                adjusted_lr = lr * (math.sqrt(1 - b2**t) / (1 - b1**t))

                p.data -= adjusted_lr * (state["first_moment"] / (torch.sqrt(state["second_moment"]) + e))
                p.data -= lr * l * p.data

                state["t"] = t + 1  # Increment iteration number.

        return loss
    
def learning_rate_scheduler(
    it, 
    max_learning_rate, 
    min_learning_rate, 
    warmup_iters, 
    cosine_cycle_iters,
):
    curr_learning_rate = None

    if it < warmup_iters:
        curr_learning_rate = (it / warmup_iters) * max_learning_rate

    elif warmup_iters <= it <= cosine_cycle_iters:
        cos_expression = np.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        curr_learning_rate = min_learning_rate + 0.5 * (1 + np.cos(cos_expression)) * (max_learning_rate - min_learning_rate)

    else:
        curr_learning_rate = min_learning_rate

    return curr_learning_rate


def gradient_clipping(parameters, max_l2_norm):
    eps = 1e-6
    total_norm = 0
    for p in parameters:
        if p.requires_grad:
            total_norm += torch.sum(torch.pow(p.grad, 2))
            
    l2_norm = torch.sqrt(total_norm)

    if l2_norm >= max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for p in parameters: 
            if p.requires_grad:
                p.grad *= scale

    return


def get_batch(dataset, batch_size, context_length, device):
    n = len(dataset)
    start_indices = torch.randint(0, n-context_length, (batch_size,))

    input_sequences = torch.empty(batch_size, context_length, dtype=torch.int64)
    token_targets = torch.empty(batch_size, context_length, dtype=torch.int64)

    for b in range(batch_size):
        start_idx = start_indices[b]
        end_idx = start_idx + context_length
        input_sequences[b, :] = torch.tensor(dataset[start_idx:end_idx])
        token_targets[b, :] = torch.tensor(dataset[start_idx+1: end_idx+1])

    return input_sequences.to(device), token_targets.to(device)


def save_checkpoint(model, optimizer, iteration, out):
    obj = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "it": iteration
    }

    torch.save(obj, out)


def load_checkpoint(src, model, optimizer):
    obj = torch.load(src)

    model.load_state_dict(obj["model_state"])
    optimizer.load_state_dict(obj["optim_state"])

    return obj["it"]


def softmax(x: torch.Tensor, dim: int, temperature: float):
    max_val = x.max(dim=dim, keepdim=True).values
    adjusted_x = x - max_val

    numerator = torch.exp(adjusted_x / temperature)
    denom = torch.sum(numerator, dim=dim).unsqueeze(dim)
    return numerator / denom


def top_p_sampling(dist, p):
    # dist -> (1, V)

    # 1. sort the distribution 
    sorted_values, sorted_idx = torch.sort(dist, dim=-1, descending=True)
    
    cum_sum = 0
    for i in range(sorted_values.shape[-1]):
        if cum_sum > p:
            break
        cum_sum += sorted_values[:,i].item()
    
    return sorted_values[:,:i] / torch.sum(sorted_values[:,:i]), sorted_idx[:,:i]




def decode(model, tok, model_input, max_generated_tokens, temperature, top_p):
    
    for i in tqdm(range(max_generated_tokens)):
        logits = model(model_input) # S, V
        last_tok_logits = logits[:,-1,:] # Grab the last one as that's all we care about


        probs = softmax(last_tok_logits, dim=-1, temperature=temperature)
        
        clipped_probs, clipped_idx = top_p_sampling(probs, top_p)

        sampled_idx = clipped_probs.multinomial(num_samples=1, replacement=True).squeeze(0).item()
        token_idx = clipped_idx[:, sampled_idx].unsqueeze(0)

        # extend the input 
        model_input = torch.cat((model_input, token_idx), dim=-1)
        
        if tok.decode(model_input[:, -1].numpy()) == '<|endoftext|>':
            break

    return tok.decode(model_input.squeeze(0).numpy())
        
        




    