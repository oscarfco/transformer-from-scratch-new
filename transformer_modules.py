import torch 
import torch.nn as nn
import numpy as np
import math
from einops import einsum, rearrange, reduce
from lora_module import LoRA

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        W = torch.empty(in_features, out_features, device=device, dtype=dtype)

        std = math.sqrt(2 / (in_features + out_features))
        W = nn.init.trunc_normal_(W, std=std, a=-3*std, b=3*std)
        self.W = nn.Parameter(W)
        
    
    def forward(self, x: torch.Tensor):
        # W is [in_features, out_features]
        return einsum(self.W, x,  "out_dim in_dim, batch seq_len in_dim -> batch seq_len out_dim")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__() 
        embedding_matrix = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        embedding_matrix = nn.init.trunc_normal_(embedding_matrix, std=1.0, a=-3.0, b=3.0)
        self.embedding_matrix = nn.Parameter(embedding_matrix)
        
    
    def forward(self, token_ids: torch.Tensor):
        return self.embedding_matrix[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps 
        gain = torch.ones(d_model, device=device, dtype=dtype)
        self.gain = nn.Parameter(gain)

    def forward(self, x: torch.Tensor):
        
        in_dtype = x.dtype
        
        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms

        return (self.gain * x).to(in_dtype)
    

class Swiglu(nn.Module):
    def __init__(self, d_ff, d_model, device=None, dtype=None):
        super().__init__()

        self.w1 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w2 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_ff, d_model, device=device, dtype=dtype)


    def silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):

        inner = einsum(self.silu(self.w1(x)), 
                       self.w3(x),
                       "... d_ff, ... d_ff -> ... d_ff") 
        
        return self.w2(inner)
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        sin_values = torch.empty((max_seq_len, d_k // 2), device=device)
        cos_values = torch.empty((max_seq_len, d_k //2), device=device)

        for i in range(max_seq_len):
            for k in range(d_k // 2):
                angle = i / (theta ** (2*k / d_k))
                sin_values[i, k] = math.sin(angle)
                cos_values[i, k] = math.cos(angle)

        self.register_buffer('sin_values', sin_values, persistent=False)
        self.register_buffer('cos_values', cos_values, persistent=False)


    def forward(self, x: torch.Tensor, token_positions=None):
        # Step 1: Slice the sin and cos tensor so we get the desired indices
        # token_positions = torch.arange(start=0, end=x.shape[1])
        # breakpoint()
        if token_positions is not None:
            token_positions = torch.arange(token_positions, device=x.device)
            sliced_sin_values = self.sin_values[token_positions]
            sliced_cos_values = self.cos_values[token_positions]
        else:
            sliced_sin_values = self.sin_values
            sliced_cos_values = self.cos_values

        even_qs = x[... , :, ::2]
        odd_qs = x[..., :, 1::2]
        
        even_final = (even_qs * sliced_cos_values) - (odd_qs * sliced_sin_values)
        odd_final = (even_qs * sliced_sin_values) + (odd_qs * sliced_cos_values)

        out = torch.empty_like(x)
        out[..., 0::2] = even_final
        out[..., 1::2] = odd_final
        return out

def softmax(x: torch.Tensor, dim: int):
    max_val = x.max(dim=dim, keepdim=True).values
    adjusted_x = x - max_val

    numerator = torch.exp(adjusted_x)
    denom = torch.sum(numerator, dim=dim).unsqueeze(dim)
    return numerator / denom


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):

    d_k = Q.shape[-1]
    attention_matrix = einsum(Q, K, "b ... seq_q d_k, b ... seq_k d_k -> b ... seq_q seq_k") / (np.sqrt(d_k))

    masked_attention_matrix = torch.where(mask, 0, float('-inf')) + attention_matrix
    masked_attention_probs = softmax(masked_attention_matrix, dim=-1)

    attention_out = einsum(masked_attention_probs, V, "b ... seq_q seq_k, b ... seq_k d_v -> b ... seq_q d_v")

    return attention_out


class multihead_self_attention(nn.Module):
    def __init__(self, d_model, num_heads, rope_params=None, lora=False, lora_config=None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = self.d_model // self.num_heads
        self.rope_params = rope_params
        self.device = device
        self.lora = lora

        if self.rope_params:
            theta = rope_params["theta"]
            max_seq_len = rope_params["max_seq_len"]
            self.token_positions = rope_params.get("token_positions", None)
            self.rope = RotaryPositionalEmbedding(theta, self.dk, max_seq_len, device=device)

        # Initialize projection weights: N(0, 2/(din + dout)) truncated at [-3σ, 3σ]
        # For d_model x d_model: σ² = 2/(d_model + d_model) = 1/d_model, so σ = 1/sqrt(d_model)
        std = math.sqrt(2 / (d_model + d_model))
        q_weight = torch.empty(d_model, d_model, device=device, dtype=dtype)
        k_weight = torch.empty(d_model, d_model, device=device, dtype=dtype)
        v_weight = torch.empty(d_model, d_model, device=device, dtype=dtype)
        o_weight = torch.empty(d_model, d_model, device=device, dtype=dtype)
        
        self.q_proj_weight = nn.Parameter(nn.init.trunc_normal_(q_weight, std=std, a=-3*std, b=3*std))
        self.k_proj_weight = nn.Parameter(nn.init.trunc_normal_(k_weight, std=std, a=-3*std, b=3*std))
        self.v_proj_weight = nn.Parameter(nn.init.trunc_normal_(v_weight, std=std, a=-3*std, b=3*std))
        self.o_proj_weight = nn.Parameter(nn.init.trunc_normal_(o_weight, std=std, a=-3*std, b=3*std))
        
        # Initialize LoRA modules for q and v projections if LoRA is enabled
        if self.lora and lora_config is not None:
            rank = lora_config.get("rank", 8)
            d = lora_config.get("d", d_model)
            self.lora_q = LoRA(rank=rank, d=d)
            self.lora_v = LoRA(rank=rank, d=d)
        
    def forward(self, x):
        Q = einsum(self.q_proj_weight, x, "hdk d_model, ... seq_len d_model -> ... seq_len hdk")
        K = einsum(self.k_proj_weight, x, "hdk d_model, ... seq_len d_model -> ... seq_len hdk")
        V = einsum(self.v_proj_weight, x, "hdk d_model, ... seq_len d_model -> ... seq_len hdk")
        
        # Add LoRA outputs to Q and V if LoRA is enabled
        if self.lora:
            Q = Q + self.lora_q(x)
            V = V + self.lora_v(x)

        # Generate the masks
        seq_len = x.shape[-2]
        mask = torch.triu(torch.ones((seq_len, seq_len), device=self.device)).T  # 1s in the lower left of the mask
        multi_head_out = []

        # Run attention for each head
        for h in range(self.num_heads):
            start_slice = h * self.dk
            end_slice = start_slice + self.dk
            sliced_q = self.rope(Q[:, :, start_slice:end_slice], seq_len) if self.rope_params else Q[:, :, start_slice:end_slice]
            sliced_k = self.rope(K[:, :, start_slice:end_slice], seq_len) if self.rope_params else K[:, :, start_slice:end_slice]
            sliced_v = V[:, :, start_slice:end_slice]

            multi_head_out.append(scaled_dot_product_attention(sliced_q, sliced_k, sliced_v, mask.bool()))
        
        multi_head_out = torch.cat(multi_head_out, dim=-1)

        return einsum(self.o_proj_weight, multi_head_out, "d_model hdv, ... seq_len hdv -> ... seq_len d_model")


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, token_positions=None, lora=False, lora_config=None, device=None):
        super().__init__()

        self.rms_norm_1 = RMSNorm(d_model=d_model, device=device)
        self.rms_norm_2 = RMSNorm(d_model=d_model, device=device)

        rope_params = {
            "theta": theta, 
            "max_seq_len": max_seq_len,
            "token_positions": token_positions
        }

        self.mha = multihead_self_attention(d_model=d_model, 
                                            num_heads=num_heads, 
                                            rope_params=rope_params,
                                            lora=lora,
                                            lora_config=lora_config,
                                            device=device)

        self.ffn = Swiglu(d_ff=d_ff, d_model=d_model, device=device)
    
    def forward(self, x):
        y = x + self.mha(self.rms_norm_1(x))

        z = y + self.ffn(self.rms_norm_2(y))

        return z
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta, token_positions=None, lora=False, lora_config=None, device=None):
        super().__init__()

        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device)
        self.num_layers = num_layers

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, 
                num_heads=num_heads, 
                d_ff=d_ff, 
                theta=rope_theta, 
                max_seq_len=context_length,
                token_positions=token_positions,
                lora=lora,
                lora_config=lora_config,
                device=device
            ) 
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model=d_model, device=device)
        self.unembedding = Linear(vocab_size, d_model, device=device)
        

    def forward(self, x):
        
        x = self.embedding(x)

        for layer_idx in range(self.num_layers):
            x = self.transformer_blocks[layer_idx](x)
        
        x = self.final_norm(x)
        x_out = self.unembedding(x)

        return x_out