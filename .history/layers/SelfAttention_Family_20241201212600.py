import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, TimerMultivariateMask, TimerCovariateMask
from einops import repeat
from layers.Attn_Projection import QueryKeyProjection, RotaryProjection
from layers.Attn_Bias import BinaryAttentionBias
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import math


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, d_model=512, num_heads=8, max_len=100, device="cuda"):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.qk_proj = QueryKeyProjection(dim=d_model, num_heads=num_heads, proj_layer=RotaryProjection, kwargs=dict(max_len=max_len),
                                          partial_factor=(0.0, 0.5),)
        # self.dim = d_model // num_heads
        # self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # self._set_cos_sin_cache(
        #     seq_len=max_len, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        # )
        self.dropout = nn.Dropout(attention_dropout)
        self.slopes = torch.Tensor(self.get_slopes(num_heads))
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        
    def apply_rope(self, q, k, position_ids):
        """
        Apply rotary position embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
            k: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
            position_ids: Position IDs tensor of shape (batch_size, seq_len)
            
        Returns:
            q_rot: Rotated query tensor
            k_rot: Rotated key tensor
        """
        # Get the frequencies for the specified positions
        # freqs = self.freqs_cis[position_ids]

        # Apply rotary embeddings
        seq_len = q.shape[1]
        q_rot, k_rot = apply_rotary_pos_emb(q, k, self.cos_cached[:seq_len].to(dtype=q.dtype),
            self.sin_cached[:seq_len].to(dtype=q.dtype), position_ids)
        
        return q_rot, k_rot
    
    def get_slopes(self,n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
        else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
            return get_slopes_power_of_2(closest_power_of_2) + self.get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]


    def get_relative_positions(self,seq_len: int,device) -> torch.tensor:
        x = torch.arange(seq_len)[None, :].to(device)
        y = torch.arange(seq_len)[:, None].to(device)
        return x - y
    
    def get_alibi_1d(self, seq_len: int, device):
        mask = self.slopes[:,None,None].to(device) * self.get_relative_positions(seq_len,device).unsqueeze(0)
        return torch.triu(mask).unsqueeze(0)
    
    def get_alibi_2d(self, n_vars: int, n_tokens: int, device):
        alibi_1d_intra = self.get_alibi_1d(n_tokens,device)
        alibi_1d_inter = torch.sqrt(1+alibi_1d_intra**2)
        intra_mask = torch.diag(torch.ones(n_tokens)).to(device)
        inter_mask = torch.triu(torch.ones((n_tokens,n_tokens))).to(device)
        mask1 = torch.einsum("mui,vj->muvij", alibi_1d_intra, intra_mask).reshape(self.num_heads,n_vars*n_tokens,n_vars*n_tokens)
        mask2 = torch.einsum("mui,vj->muvij", alibi_1d_inter, inter_mask).reshape(self.num_heads,n_vars*n_tokens,n_vars*n_tokens)
        return (mask1+mask2).unsqueeze(0)

    def forward(self, queries, keys, values, attn_mask, n_vars=None, n_tokens=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        
        if n_vars is not None and n_tokens is not None:
            seq_id = repeat(torch.arange(n_tokens), 'n -> c n', c=n_vars).reshape(-1)
            seq_id = repeat(seq_id, 'n -> b h n', b=B, h=H)
            
            # self.apply_rope(queries, keys, seq_id)
            # queries, keys = self.qk_proj(
            #     queries, keys, query_id=seq_id, kv_id=seq_id)
            alibi = self.get_alibi_2d(n_vars, n_tokens, queries.device)
        else:
            alibi = self.get_alibi_1d(n_tokens, queries.device)

        scores = torch.einsum("bhle,bhse->bhls", queries, keys) + alibi

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class TimeAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, d_model=512, num_heads=8, max_len=100, covariate=False, flash_attention=False):
        super(TimeAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.covariate = covariate
        self.flash_attention = flash_attention
        self.qk_proj = QueryKeyProjection(dim=d_model, num_heads=num_heads, proj_layer=RotaryProjection, kwargs=dict(max_len=max_len),
                                          partial_factor=(0.0, 0.5),)
        self.attn_bias = BinaryAttentionBias(dim=d_model, num_heads=num_heads)

    def forward(self, queries, keys, values, attn_mask, n_vars, n_tokens, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # [B, H, L, E]
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:
            values = values.permute(0, 2, 1, 3)

        seq_id = torch.arange(n_tokens * n_vars)
        # seq_id = repeat(torch.arange(n_tokens), 'n -> c n', c=n_vars).reshape(-1)
        seq_id = repeat(seq_id, 'n -> b h n', b=B, h=H)

        queries, keys = self.qk_proj(
            queries, keys, query_id=seq_id, kv_id=seq_id)

        scale = self.scale or 1. / sqrt(E)

        var_id = repeat(torch.arange(n_vars),
                        'C -> (C n_tokens)', n_tokens=n_tokens)
        var_id = repeat(var_id, 'L -> b h L', b=B, h=1).to(queries.device)

        attn_bias = self.attn_bias(var_id, var_id)

        if self.mask_flag:
            if attn_mask is None:
                if self.covariate:
                    attn_mask = TimerCovariateMask(
                        B, n_vars, n_tokens, device=queries.device)
                else:
                    attn_mask = TimerMultivariateMask(
                        B, n_vars, n_tokens, device=queries.device)
            attn_mask = attn_bias.masked_fill(attn_mask.mask, float("-inf"))
        else:
            attn_mask = attn_bias

        if self.flash_attention:
            V = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values, attn_mask)
        else:
            scores = torch.einsum("bhle,bhse->bhls", queries, keys)
            scores += attn_mask
            
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, n_vars=None, n_tokens=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            n_vars=n_vars,
            n_tokens=n_tokens,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
