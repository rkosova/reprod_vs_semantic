import torch
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import DataLoader

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import math


from functools import partial
from tqdm import tqdm

from rvsablation.jepa import FSQ_V_JEPA, PredictorTransformer, EMA
from rvsablation.utils import generate_multi_block_mask, seed_everything, read_config, write_config, fsq_2_index, get_dataset

import os
# import sys
# from pathlib import Path
# # just for now using this
# parent_dir = str(Path('..').resolve())
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)
# from dmc_dataset.dataset import get_dataset


def _get_uniform_std_target(fsq_levels):
    """
    Given FSQ levels per dimension, the STD target is computed by making the assumption
    of discrete uniform distributions of embedding dimensions with range `- fsq_levels // 2` to `fsq_levels // 2`
    matching the binding function. The STD target is then computed by using the formula for the variance
    of a discrete uniform distribution per dimension. https://en.wikipedia.org/wiki/Discrete_uniform_distribution
    """
    max_val = fsq_levels // 2
    min_val = -max_val

    var = ((max_val - min_val + 1) ** 2 - 1) / 12

    return var ** 0.5

def _lr_lambda(epoch, total_epochs, warmup_epochs):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs # linear warmup, epoch starts at 1
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs) # post warmup epochs passed / post warmup total epochs
    return 0.5 * (1 + math.cos(math.pi * progress))

def _ema_momentum(step, total_steps, ema_starting_momentum, ema_final_momentum):
    if total_steps <= 1:
        return ema_final_momentum
    
    progress = step / (total_steps - 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return ema_final_momentum - (ema_final_momentum - ema_starting_momentum) * cosine # cos schedule from starting to final momentum

def _variance_hinge_loss(z, target_std, eps=1e-4):
    std = torch.sqrt(z.float().var(dim=(0, 1)) + eps)
    return F.relu(target_std - std).mean()


def _train_epoch(
        model: FSQ_V_JEPA,
        predictor: PredictorTransformer,
        teacher: EMA,
        dataloader: DataLoader,
        optim: optim.AdamW,
        device: str,
        radii: torch.Tensor,
        levels_max: torch.Tensor,
        global_step,
        ema_scheduler,
        var_std_target,
        var_reg=0.0,
        short_range_targets=2,
        long_range_targets=2,
        short_scale=(.1, .15),
        short_ar=(.75, 1.5),
        long_scale=(.2, .35),
        long_ar=(.35, 1.5),
        context_scale=(1., 1.),
        context_ar=(1., 1.)
    ):
    epoch_loss = 0.0

    utilized_indices = []    
    
    bar = tqdm(enumerate(dataloader), desc="Training", total=len(dataloader))

    for _, (X, *_) in bar:
        optim.zero_grad(set_to_none=True)
        X = X.to(device, non_blocking=True)

        with torch.autocast(device, dtype=torch.bfloat16):
            ctx_index, tgt_index_list = generate_multi_block_mask(
                spatial_grid_size=model.encoder.spatial_grid_size, temporal_grid_size=model.encoder.temporal_grid_size, 
                short_range_targets=short_range_targets, long_range_targets=long_range_targets,
                short_scale=short_scale, long_scale=long_scale, short_ar=short_ar, long_ar=long_ar,
                context_scale=context_scale, context_ar=context_ar
            )

            Z_masked = model(X, ctx_index)
            Z_target_pred, target_index = predictor(Z_masked, ctx_index, tgt_index_list)
            Z_target = teacher.target(X, quantized=False)[:, target_index, :]
            Z_target = Z_target.detach()

            loss = F.l1_loss(Z_target_pred, Z_target)
            if var_reg > 0:
                var_loss = var_reg * _variance_hinge_loss(Z_masked, var_std_target)
                loss = loss + var_loss
        
        loss.backward()
        optim.step()
        teacher.momentum = ema_scheduler(global_step)
        teacher.update(model)
        global_step += 1


        batch_utilized_indices = fsq_2_index(Z_masked.reshape(-1, Z_masked.shape[-1]), radii, levels_max)
        utilized_indices.append(batch_utilized_indices.unique())
    
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    utilized_indices = torch.cat(utilized_indices).unique()

    return avg_loss, utilized_indices.shape[0], global_step

def train(
        epochs=30,
        warmup_epochs=10,
        batch_size=400,
        lr=2e-4,
        run_name="fsq_v_jepa_run",
        seed=42,
        data_num_traj=25000,
        data_num_frames=100,
        sample_num_frames=8,
        sample_num_clips=12,
        data_path="./data/walker_trajs_h5/data.h5",
        stride=1,
        ema_starting_momentum=0.996,
        ema_final_momentum=0.999,
        var_reg=0.0,
        short_range_targets=2,
        long_range_targets=2,
        short_scale=(.1, .15),
        short_ar=(.75, 1.5),
        long_scale=(.2, .35),
        long_ar=(.35, 1.5),
        context_scale=(1., 1.),
        context_ar=(1., 1.)
    ):

    RUN_PARAMS = { # saving to dics so that they may be easily saved later on
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "run_name": run_name,
        "seed": seed,
        "data_num_traj": data_num_traj,
        "data_num_frames": data_num_frames,
        "sample_num_frames": sample_num_frames,
        "sample_num_clips": sample_num_clips,
        "data_path": data_path,
        "stride": stride,
        "ema_starting_momentum": ema_starting_momentum,
        "ema_final_momentum": ema_final_momentum,
        "var_reg": var_reg,
        "short_range_targets": short_range_targets,
        "long_range_targets": long_range_targets,
        "short_scale": short_scale,
        "short_ar": short_ar,
        "long_scale": long_scale,
        "long_ar": long_ar,
        "context_scale": context_scale,
        "context_ar": context_ar
    }


    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 STARTING TRAINING ON {DEVICE} 🚀")
    rng = seed_everything(seed=seed)

    # DATASET HANDLING
    shift_transform = T.Compose([
        T.Pad(padding=10, padding_mode='edge'),
        T.RandomCrop(size=(84, 84))
    ])

    dataset, testset, *_ = get_dataset(
        rng=rng,
        root=data_path,
        sample_num_frames=sample_num_frames,
        sample_num_clips=sample_num_clips,
        data_num_traj=data_num_traj,
        data_num_frames=data_num_frames,
        stride=stride,
        train_transform=shift_transform,
        dataset=data_path.split("/")[2].split("_")[0]
    )


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #######################################################

    # MODEL, TEACHER, PREDICTOR HANDLING
    model_config = read_config("./jepa_conf/jepa_model_config.json")
    model = FSQ_V_JEPA(**model_config)
    model = model.to(DEVICE)
    
    teacher = EMA(student=model, momentum=ema_starting_momentum)

    predictor_config = read_config("./jepa_conf/jepa_predictor_config.json")
    predictor = PredictorTransformer(
        spatial_grid_size=model.encoder.spatial_grid_size,
        temporal_grid_size=model.encoder.temporal_grid_size,
        input_dim=model_config["fsq_dim"],
        output_dim=model_config["fsq_dim"],
        **predictor_config
    )
    predictor = predictor.to(DEVICE)

    model_p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pred_p_count  = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"Loaded FSQ-V-JEPA with {model_p_count:,} parameters.")
    print(f"Loaded PredictorTransformer with {pred_p_count:,} parameters.")
    p_count = model_p_count / 1_000_000
    pp_count = pred_p_count / 1_000_000
    #######################################################


    # SETTING UP FSQ UTILIZATION TRACKING
    levels = model_config["fsq_levels"]
    levels_max = levels // 2
    levels_max = levels_max.reshape(-1)
    levels_corrected = (levels_max * 2) + 1 # (fsq_dim,)

    radii = torch.flip(levels_corrected, dims=(-1,))
    radii = torch.cumprod(radii, dim=-1) / radii
    radii = torch.flip(radii, dims=(-1,)) # (fsq_dim, )

    levels_max = levels_max.to(DEVICE)
    radii = radii.to(DEVICE)
    #######################################################

    # HANDLING VARIANCE LOSS
    var_std_target = _get_uniform_std_target(fsq_levels=levels).squeeze(0) # (1, fsq_dim)
    var_std_target = var_std_target.to(DEVICE)
    #######################################################
    
    # HANDLING WEIGHT DECAY
    decay_params = []
    no_decay_params = []
    for name, param in list(model.named_parameters()) + list(predictor.named_parameters(prefix="predictor")):
        if not param.requires_grad:
            continue
        if (
            name.startswith("quantizer.")
            or name.endswith(".bias")
            or "norm" in name.lower()
        ): # no decay for quantizer (gain and projection), biases, or norms (layernorm only here)
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    trainable_params = [
        {"params": decay_params, "weight_decay": 0.05},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    #######################################################

    # OPTIMIZATION
    optimizer = optim.AdamW(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.95)
    )

    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=partial(_lr_lambda, total_epochs=epochs, warmup_epochs=warmup_epochs))
    #######################################################

    # EMA SCHEDULING
    ema_scheduler = partial(_ema_momentum, 
                            total_steps=epochs * len(dataloader),
                            ema_starting_momentum=ema_starting_momentum,
                            ema_final_momentum=ema_final_momentum
                        )
    global_step = 0
    #######################################################

    model.train()
    predictor.train()

    for epoch in range(1, epochs + 1):
        avg_loss, epoch_util, global_step = _train_epoch(
            model=model,
            predictor=predictor,
            teacher=teacher,
            dataloader=dataloader,
            optim=optimizer,
            device=DEVICE,
            radii=radii[None, :], # (1, fsq_dim)
            levels_max=levels_max[None, :], # (1, fsq_dim)
            global_step=global_step,
            ema_scheduler=ema_scheduler,
            var_std_target=var_std_target,
            var_reg=var_reg,
            short_range_targets=short_range_targets, long_range_targets=long_range_targets,
            short_scale=short_scale, long_scale=long_scale, short_ar=short_ar, long_ar=long_ar,
            context_scale=context_scale, context_ar=context_ar
        )

        print(f"Epoch: [{epoch}/{epochs}] | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Momentum: {teacher.momentum:.5f}")
        print(f"FSQ Utilization over the epoch: [{epoch_util}] unique codes.")
        scheduler.step()

    os.makedirs(f"./jepa_runs/{run_name}", exist_ok=True)
    os.makedirs(f"./jepa_runs/{run_name}/weights", exist_ok=True)
    os.makedirs(f"./jepa_runs/{run_name}/confs", exist_ok=True)

    torch.save(model.state_dict(), f"./jepa_runs/{run_name}/weights/model_{p_count:.1f}m.pt")
    torch.save(predictor.state_dict(), f"./jepa_runs/{run_name}/weights/predictor_{pp_count:.2f}m.pt")
    torch.save(teacher.teacher.state_dict(), f"./jepa_runs/{run_name}/weights/teacher_{p_count:.2f}m.pt")

    write_config(f"./jepa_runs/{run_name}/confs/model_config.json", model_config)
    write_config(f"./jepa_runs/{run_name}/confs/predictor_config.json", predictor_config)
    write_config(f"./jepa_runs/{run_name}/confs/run_config.json", RUN_PARAMS)

    print(f"✅ Training Complete! Weights saved to jepa_runs/{run_name}")