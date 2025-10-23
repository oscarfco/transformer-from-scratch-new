import torch 
import torch.nn as nn
import numpy as np
import math
from einops import einsum, rearrange, reduce

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        W = torch.empty(in_features, out_features, device=device, dtype=dtype)
        W = nn.init.trunc_normal_(W)
        self.W = nn.Parameter(W)
        
    
    def forward(self, x: torch.Tensor):
        # W is [in_features, out_features]
        return einsum(self.W, x,  "out_dim in_dim, batch seq_len in_dim -> batch seq_len out_dim")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__() 
        embedding_matrix = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        embedding_matrix = nn.init.trunc_normal_(embedding_matrix)
        self.embedding_matrix = nn.Parameter(embedding_matrix)
        
    
    def forward(self, token_ids: torch.Tensor):
        return self.embedding_matrix[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps 
        gain = torch.empty(d_model, device=device, dtype=dtype)
        gain = nn.init.trunc_normal_(gain)
        self.gain = nn.Parameter(gain)

    def forward(self, x: torch.Tensor):
        
        in_dtype = x.dtype
        x = x.to(torch.float32)

        denom = torch.sqrt((1/self.d_model) * reduce(x**2 + self.eps, "... d_model -> ... ", 'sum'))
        denom = rearrange(denom, "... -> ... 1")

        result =  einsum(torch.div(x, denom), self.gain, "... d_model, d_model -> ... d_model")
        return result.to(in_dtype)
    

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
        if token_positions is not None:
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
    def __init__(self, d_model, num_heads, rope_params=None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = self.d_model // self.num_heads
        self.rope_params = rope_params
        self.device = device

        if self.rope_params:
            theta = rope_params["theta"]
            max_seq_len = rope_params["max_seq_len"]
            self.token_positions = rope_params.get("token_positions", None)
            self.rope = RotaryPositionalEmbedding(theta, self.dk, max_seq_len, device=device)

        self.q_proj_weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(d_model, d_model, device=device, dtype=dtype)))
        self.k_proj_weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(d_model, d_model, device=device, dtype=dtype)))
        self.v_proj_weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(d_model, d_model, device=device, dtype=dtype)))
        self.o_proj_weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(d_model, d_model, device=device, dtype=dtype)))
        
    def forward(self, x):
        Q = einsum(self.q_proj_weight, x, "hdk d_model, ... seq_len d_model -> ... seq_len hdk")
        K = einsum(self.k_proj_weight, x, "hdk d_model, ... seq_len d_model -> ... seq_len hdk")
        V = einsum(self.v_proj_weight, x, "hdk d_model, ... seq_len d_model -> ... seq_len hdk")

        # Generate the masks
        seq_len = x.shape[-2]
        mask = torch.triu(torch.ones((seq_len, seq_len), device=self.device)).T  # 1s in the lower left of the mask
        multi_head_out = []

        # Run attention for each head
        for h in range(self.num_heads):
            start_slice = h * self.dk
            end_slice = start_slice + self.dk
            sliced_q = self.rope(Q[:, :, start_slice:end_slice], self.token_positions) if self.rope_params else Q[:, :, start_slice:end_slice]
            sliced_k = self.rope(K[:, :, start_slice:end_slice], self.token_positions) if self.rope_params else K[:, :, start_slice:end_slice]
            sliced_v = V[:, :, start_slice:end_slice]

            multi_head_out.append(scaled_dot_product_attention(sliced_q, sliced_k, sliced_v, mask.bool()))
        
        multi_head_out = torch.cat(multi_head_out, dim=-1)

        return einsum(self.o_proj_weight, multi_head_out, "d_model hdv, ... seq_len hdv -> ... seq_len d_model")


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, token_positions=None, device=None):
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
                                            device=device)

        self.ffn = Swiglu(d_ff=d_ff, d_model=d_model, device=device)
    
    def forward(self, x):
        y = x + self.mha(self.rms_norm_1(x))

        z = y + self.ffn(self.rms_norm_2(y))

        return z
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta, token_positions=None, device=None):
        super().__init__()

        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device)
        self.num_layers = num_layers

        self.transformer_blocks = [
            TransformerBlock(
                d_model=d_model, 
                num_heads=num_heads, 
                d_ff=d_ff, 
                theta=rope_theta, 
                max_seq_len=context_length,
                token_positions=token_positions,
                device=device
            ) 
            for _ in range(num_layers)
        ]

        self.final_norm = RMSNorm(d_model=d_model, device=device)
        self.unembedding = Linear(vocab_size, d_model, device=device)
        

    def forward(self, x):
        
        x = self.embedding(x)

        for layer_idx in range(self.num_layers):
            x = self.transformer_blocks[layer_idx](x)
        
        x = self.final_norm(x)
        x_out = self.unembedding(x)

        return x_out