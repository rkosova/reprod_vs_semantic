import torch
from torch import nn
import torch.nn.functional as F

from .core import _TransformerStack, FSQ, ViViT
from .utils import get_3d_sincos_pos_embed

class FSQ_VideoVAE(nn.Module):
    def __init__(self, 
                 image_size,
                 duration,
                 patch_size,
                 tubelet_size,
                 encoder_dim,
                 encoder_depth,
                 attention_dim,
                 attention_num_heads,
                 fsq_dim,
                 fsq_levels,
                 causal=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = ViViT(
            image_size=image_size,
            duration=duration,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            encoder_dim=encoder_dim,
            encoder_depth=encoder_depth,
            attention_dim=attention_dim,
            attention_num_heads=attention_num_heads,
            causal=causal
        )

        self.quantizer = FSQ(
            dim=encoder_dim,
            fsq_dim=fsq_dim,
            levels=fsq_levels
        )


    def encode(self, x):
        return self.encoder(x, ctx_idx=None)

    def quantize(self, x, quantized, use_gain):
        return self.quantizer(x) if quantized else self.quantizer.bind(x, use_gain=use_gain)

    def forward(self, x, quantized=True, use_gain=True):
        # quantized and use_gain are left as inference parameters
        # so as to allow the teacher to deviate from the student along these
        x = self.encode(x)
        x = self.quantize(x, quantized=quantized, use_gain=use_gain)
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, 
                 spatial_grid_size,
                 temporal_grid_size,
                 input_dim,
                 decoder_dim,
                 output_dim,
                 depth,
                 attention_dim,
                 attention_num_heads,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.in_proj = nn.Linear(input_dim, decoder_dim)
        self.stack = _TransformerStack(decoder_dim, depth, attention_dim, attention_num_heads)

        pos = get_3d_sincos_pos_embed(decoder_dim, temporal_grid_size, spatial_grid_size)
        self.register_buffer(
            "pos",
            pos.unsqueeze(0),
            persistent=False
        )

        self.out_proj = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim * 4),
            nn.GELU(),
            nn.Linear(4 * decoder_dim, output_dim)
        )

    def forward(self, x):
        x = self.in_proj(x)
        x = x + self.pos
        x = self.stack(x)
        x = self.out_proj(x)

        return x