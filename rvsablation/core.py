import torch
from torch import nn
import torch.nn.functional as F
from .utils import patchify_clip, get_3d_sincos_pos_embed, jepa_mask


class _TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 attn_dim,
                 num_heads,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert attn_dim % num_heads == 0
        assert (attn_dim // num_heads) % 8 == 0

        self.attn_dim = attn_dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim, bias=False)
        self.qkv = nn.Linear(dim, 3 * attn_dim)
        self.attn_proj = nn.Linear(attn_dim, dim)
        self.norm2 = nn.LayerNorm(dim, bias=False)

        self.lin1 = nn.Linear(dim, 4 * dim)
        self.ac1 = nn.GELU()
        self.lin2 = nn.Linear(4 * dim, dim)

    def _attention(self, x, mask=None):
        B, N, _ = x.shape
        A = self.attn_dim
        Nh = self.num_heads
        Dh = self.attn_dim // self.num_heads

        QKV = self.qkv(x) # B, N, 3A
        QKV = QKV.reshape(B, N, 3, Nh, Dh)
        QKV = QKV.permute(2, 0, 3, 1, 4) # 3, B, Nh, N, Dh
        Q, K, V = QKV.unbind(0) # Q, K, V: B, Nh, N, Dh

        y = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask) # B, Nh, N, Dh
        y = y.permute(0, 2, 1, 3) # B, N, Nh, Dh
        y = y.reshape(B, N, A)

        return self.attn_proj(y)
    
    def forward(self, x, attn_mask=None):
        y = self.norm1(x)
        y = self._attention(y, attn_mask)
        x = x + y

        y = self.norm2(x)
        y = self.lin1(y)
        y = self.ac1(y)
        y = self.lin2(y)

        return x + y
    

class _TransformerStack(nn.Module):
    def __init__(self, 
                 dim, 
                 depth, 
                 attn_dim, 
                 num_heads, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.blocks = nn.ModuleList([
            _TransformerBlock(dim, attn_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim, bias=False)

    def forward(self, x, attn_mask=None):
        for block in self.blocks:
            x = block(x, attn_mask)

        return self.norm(x)
    

class FSQ(nn.Module):
    def __init__(self, 
                 dim,
                 fsq_dim,
                 levels,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        

        self.register_buffer(
            "levels",
            levels,
            persistent=False
        ) # handling cuda

        self.gain = nn.Parameter(torch.tensor(1.0))
        self.proj = nn.Linear(dim, fsq_dim, bias=False)
        self.f = lambda z: (self.levels // 2) * F.tanh(z)

    def compress(self, x, use_gain=True):
        out = self.proj(x)
        return out * self.gain if use_gain else out
    
    def bind(self, x, use_gain=True):
        return self.f(self.compress(x, use_gain))
    
    def forward(self, x, use_gain=True):
        x = self.bind(x, use_gain)
        x = x + (torch.round(x) - x).detach()

        return x


class ViViT(nn.Module):
    """
    Can be used for both the AE paradigm (no masking indices are passed) as well as the predictive/JEPA paradigm (sampled ctx masking indices are passed).
    """
    def __init__(self, 
                 image_size,
                 duration,
                 patch_size,
                 tubelet_size,
                 encoder_dim,
                 encoder_depth,
                 attention_dim,
                 attention_num_heads,
                 causal=False,
                 *args, **kwargs):
        """
        Simple Video ViT encoder which supports masking (through indices) and temporal-causal (block lower-triangular) causal attention.

        Uses fixed 3D spatio-temporal sin-cos positional embeddings.
        """
        super().__init__(*args, **kwargs)
        
        assert image_size % patch_size == 0, f"Image of `size` {image_size}x{image_size} can not be broken down into non-overlapping patches of {patch_size}x{patch_size}. `patch_size` must divide `image_size`!"
        assert duration % tubelet_size == 0, f"Clip of `duration` {duration} frames cannot be broken down into non-overlapping tubelets of {tubelet_size}. `tubelet_size` must divide `duration`!"

        self.image_size = image_size
        self.duration = duration
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self.spatial_grid_size = image_size // patch_size
        self.temporal_grid_size = duration // tubelet_size

        self.tubelet_dim = self.tubelet_size * (self.patch_size ** 2) * 3 # color 

        self.tubelet_proj = nn.Linear(self.tubelet_dim, encoder_dim)
        
        pos = get_3d_sincos_pos_embed(encoder_dim, self.temporal_grid_size, self.spatial_grid_size) # (N, encoder_dim)
        self.register_buffer(
            "pos",
            pos.unsqueeze(0), # unsqueezed for broadcasted addition
            persistent=False
        )

        self.causal = causal

        tokens_per_time = self.spatial_grid_size ** 2
        full_temporal_ids = torch.arange(
            self.temporal_grid_size
        ).repeat_interleave(tokens_per_time)
        
        _temporal_attn_mask = full_temporal_ids[:, None] >= full_temporal_ids[None, :]

        self.register_buffer(
            "_temporal_attn_mask",
            _temporal_attn_mask,
            persistent=False
        ) # to handle device

        self.stack = _TransformerStack(encoder_dim, encoder_depth, attention_dim, attention_num_heads)

    def _get_attn_mask(self, ctx_idx):
        if not self.causal: # if not causal we dont need an attention mask
            return None
        
        mask = self._temporal_attn_mask # if causal, but not masked, we use the full temporal attention mask

        if ctx_idx is not None: 
            mask = mask[ctx_idx][:, ctx_idx]# if causal and masked, we index the rows and columns of the full attn. mask corresponding to the context positions
                                            # rows correspond to queries and columns to keys
        return mask 
    
    def forward(self, x, ctx_idx=None):
        x = patchify_clip(x, self.tubelet_size, self.patch_size)
        x = self.tubelet_proj(x)
        x = x + self.pos

        if ctx_idx is not None:
            x = jepa_mask(x, ctx_idx) # just gathers

        attn_mask = self._get_attn_mask(ctx_idx)
        x = self.stack(x, attn_mask)
        return x

