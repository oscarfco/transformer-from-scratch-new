'''
Implementing LoRA from scratch. 

1. we start with a preinitialized model
2. For each weight matrix of interest, we want to construct two low rank matricies A and B
    (i). Following Hu et al. I will just apply LoRA to W_q and W_v. 
    (ii). We want to have rank as a parameter


**Method**

- Each attention module will have a LoRA class (if enabled) that will store A and B 
- A is trunc_normal_ intialized and B is zeros s.t. B@A=0 at the start. 

- have a global lora boolean arg which if true alters how we do the attention forward pass
- along with lora arg we also pass a lora_config dict to the model. 
- if lora is enabled we must also freeze all the parameters not of interest. 
'''

import torch 
import torch.nn as nn
import math
from einops import einsum

class LoRA(nn.Module):
    def __init__(self, rank, d):
        '''
        rank: an int for how large the LoRA rank it
        d: the other dimension size (for attn modules typically d_model)
        '''

        super().__init__()

        self.rank = rank
        self.d = d

        std = math.sqrt(2 / (rank + d))
        self.A = nn.Parameter(nn.init.trunc_normal_(torch.empty(rank, d), std=std, a=-3*std, b=3*std))
        self.B = nn.Parameter(torch.zeros((d, rank)))
    
    def forward(self, x):
       
        a_x = einsum(self.A, x, "rank d_model, b s d_model -> b s rank")
        return einsum(self.B, a_x, "d_model rank, b s rank -> b s d_model")

