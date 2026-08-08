import torch
from torch import nn
import torch.nn.functional as F

from .core import _TransformerStack, FSQ, ViViT
from .utils import get_3d_sincos_pos_embed, build_predictor_input

from copy import deepcopy

class FSQ_V_JEPA(nn.Module):
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

    def encode(self, x, ctx_idx):
        return self.encoder(x, ctx_idx)

    def quantize(self, x, quantized, use_gain):
        return self.quantizer(x) if quantized else self.quantizer.bind(x, use_gain=use_gain)

    def forward(self, x, ctx_idx=None, quantized=True, use_gain=True):
        # quantized and use_gain are left as inference parameters
        # so as to allow the teacher to deviate from the student along these
        x = self.encode(x, ctx_idx=ctx_idx)
        x = self.quantize(x, quantized=quantized, use_gain=use_gain)

        return x
    

class PredictorTransformer(nn.Module):
    def __init__(self,
                 spatial_grid_size,  # grid sizes as parameters instead of computed since this is an auxillary model so by the time its constructed
                 temporal_grid_size, # we should already have access to these through ViViT
                 input_dim,
                 predictor_dim,
                 output_dim,
                 depth,
                 attention_dim,
                 attention_num_heads, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.in_proj = nn.Linear(input_dim, predictor_dim)
        self.stack = _TransformerStack(predictor_dim, depth, attention_dim, attention_num_heads)
        self.mask_token = nn.Parameter(torch.rand(predictor_dim) * 0.02) # .expand in build_predictor_input handles necessary unsqueezing

        pos = get_3d_sincos_pos_embed(predictor_dim, temporal_grid_size, spatial_grid_size) # (N, predictor_dim)
        self.register_buffer(
            "pos",
            pos, # .expand in build_predictor_input handles necessary unsqueezing
            persistent=False
        )

        self.out_proj = nn.Sequential(
            nn.Linear(predictor_dim, predictor_dim * 4),
            nn.GELU(),
            nn.Linear(4 * predictor_dim, output_dim)
        )

    def forward(self, x, ctx_idx, tgt_idx_list):
        x = self.in_proj(x)
        x, full_idx, Kc = build_predictor_input(x, ctx_idx, tgt_idx_list, self.pos, self.mask_token)
        x = self.stack(x)
        x = self.out_proj(x)

        target_idx = full_idx[Kc:] # full_idx: (B, Kc + Kt,) => target_idx: (B, Kt)
        pred = x[:, Kc:, :] # x: (B, Kc + Kt, D) => pred: (B, Kt, D)

        return pred, target_idx
    

class EMA:
    def __init__(self,
                 student,
                 momentum=0.996):
        self.momentum = momentum
        self.teacher = deepcopy(student)
        self.teacher.eval()

        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def target(self, x, quantized=False, use_gain=True):
        return self.teacher(x.detach(), ctx_idx=None, quantized=quantized, use_gain=use_gain)
    
    @torch.no_grad()
    def update(self, student):
        for teacher_param, student_param in zip(
            self.teacher.parameters(),
            student.parameters()
        ):
            # inplace ops so that the actual parameters are changed
            # teacher_param is a reference to the actual tensor objects
            teacher_param.mul_(self.momentum)
            teacher_param.add_(
                student_param.detach(),
                alpha=1.0 - self.momentum
            )

        for teacher_buffer, student_buffer in zip(
            self.teacher.buffers(),
            student.buffers()
        ):
            teacher_buffer.copy_(student_buffer)