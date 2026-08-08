import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import math
import random

from functools import partial
from tqdm import tqdm

from rvsablation.ae import FSQ_VideoVAE, DecoderTransformer
from rvsablation.utils import seed_everything, read_config, write_config, fsq_2_index, patchify_clip, unpatchify_clip, get_dataset


import os
# import sys
# from pathlib import Path
# # just for now using this
# parent_dir = str(Path('..').resolve())
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)
# from dmc_dataset.dataset import get_dataset

def _lr_lambda(epoch, total_epochs, warmup_epochs):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs # linear warmup, epoch starts at 1
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs) # post warmup epochs passed / post warmup total epochs
    return 0.5 * (1 + math.cos(math.pi * progress)) # cos schedule from lr to 0


def _train_epoch(
        model: FSQ_VideoVAE,
        decoder: DecoderTransformer,
        dataloader: DataLoader,
        optimizer: optim.AdamW,
        device: str,
        radii: torch.Tensor,
        levels_max: torch.Tensor,
    ):

    epoch_loss = 0.0

    utilized_indices = []

    bar = tqdm(enumerate(dataloader), desc="Training", total=len(dataloader))

    for _, (X, _, _, _, _, _) in bar:
        optimizer.zero_grad(set_to_none=True)
        X = X.to(device)

        with torch.autocast(device, dtype=torch.bfloat16):
            target_X = patchify_clip(X,
                                     tubelet_size=model.encoder.tubelet_size,
                                     patch_size=model.encoder.patch_size)
            Z = model(X)
            pred_X = decoder(Z)
            loss = F.mse_loss(pred_X, target_X)

        loss.backward()
        optimizer.step()

        batch_utilized_indices = fsq_2_index(Z.reshape(-1, Z.shape[-1]), radii, levels_max)
        utilized_indices.append(batch_utilized_indices.unique())

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    utilized_indices = torch.cat(utilized_indices).unique()
    
    return avg_loss, utilized_indices.shape[0]

@torch.no_grad()
def _eval_epoch(
        model: FSQ_VideoVAE,
        decoder: DecoderTransformer,
        dataloader: DataLoader,
        device: str,
        radii,
        levels_max,
    ):

    epoch_loss = 0.0

    utilized_indices = []
    
    bar = tqdm(enumerate(dataloader), desc=" ↪Eval.", total=len(dataloader))

    for _, (X, _, _, _, _, _) in bar:
        X = X.to(device)

        with torch.autocast(device, dtype=torch.bfloat16):
            target_X = patchify_clip(X,
                                     tubelet_size=model.encoder.tubelet_size,
                                     patch_size=model.encoder.patch_size)
            Z = model(X)
            pred_X = decoder(Z)
            loss = F.mse_loss(pred_X, target_X)


        batch_utilized_indices = fsq_2_index(Z.reshape(-1, Z.shape[-1]), radii, levels_max)
        utilized_indices.append(batch_utilized_indices.unique())

        epoch_loss += loss.item()

    random_idx = random.randint(0, pred_X.shape[0] - 1)
    save_x = pred_X[random_idx].unsqueeze(0).cpu() # (N, D) -> (1, N, D)
    save_x = unpatchify_clip(save_x, duration=model.encoder.duration, tubelet_size=model.encoder.tubelet_size, 
                                     image_size=model.encoder.image_size, patch_size=model.encoder.patch_size)

    utilized_indices = torch.cat(utilized_indices).unique()

    return epoch_loss / len(dataloader), utilized_indices.shape[0], save_x.squeeze(0)

def train(
        epochs=30,
        warmup_epochs=10,
        batch_size=400,
        lr=2e-4,
        run_name="fsq_videovae_run",
        seed=42,
        data_num_traj=25000,
        data_num_frames=100,
        sample_num_frames=8,
        sample_num_clips=12
):
    RUN_PARAMS = {
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
    }

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 STARTING TRAINING ON {DEVICE} 🚀")
    rng = seed_everything(seed=seed)

    # DATASET HANDLING
    dataset, testset, _, _ = get_dataset(
        rng=rng,
        root="./data/walker_trajs_h5/data.h5",
        sample_num_frames=sample_num_frames,
        sample_num_clips=sample_num_clips,
        data_num_traj=data_num_traj,
        data_num_frames=data_num_frames,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #######################################################

    # MODEL AND DECODER
    model_config = read_config("./ae_conf/ae_model_config.json")
    model = FSQ_VideoVAE(**model_config)
    model = model.to(DEVICE)

    decoder_config = read_config("./ae_conf/ae_decoder_config.json")
    decoder = DecoderTransformer(
        spatial_grid_size=model.encoder.spatial_grid_size,
        temporal_grid_size=model.encoder.temporal_grid_size,
        input_dim=model_config["fsq_dim"],
        output_dim=model.encoder.tubelet_dim,
        **decoder_config
    )
    decoder = decoder.to(DEVICE)

    model_p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dec_p_count  = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Loaded FSQ-V-JEPA with {model_p_count:,} parameters.")
    print(f"Loaded DecoderTransformer with {dec_p_count:,} parameters.")
    p_count = model_p_count / 1_000_000
    dp_count = dec_p_count / 1_000_000
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

    # HANDLING WEIGHT DECAY
    decay_params = []
    no_decay_params = []
    for name, param in list(model.named_parameters()) + list(decoder.named_parameters(prefix="decoder")):
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

    model.train()
    decoder.train()

    os.makedirs(f"./ae_runs/{run_name}", exist_ok=True)
    os.makedirs(f"./ae_runs/{run_name}/weights", exist_ok=True)
    os.makedirs(f"./ae_runs/{run_name}/confs", exist_ok=True)
    os.makedirs(f"./ae_runs/{run_name}/training_outputs", exist_ok=True)

    for epoch in range(1, epochs + 1):
        avg_loss, epoch_util = _train_epoch(
            model=model,
            decoder=decoder,
            dataloader=dataloader,
            optimizer=optimizer,
            device=DEVICE,
            radii=radii[None, :],
            levels_max=levels_max[None, :]
        )
        print(f"Epoch: [{epoch}/{epochs}] | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"FSQ Utilization over the epoch: [{epoch_util}] unique codes.")
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1: # short circuiting ftw
            avg_eval_loss, eval_epoch_util, save_x = _eval_epoch(
                model=model,
                decoder=decoder,
                dataloader=testloader,
                device=DEVICE,
                radii=radii[None, :],
                levels_max=levels_max[None, :]
            )
            save_image(save_x, f"./ae_runs/{run_name}/training_outputs/epoch_{epoch}.png", nrow=model_config["duration"]) # nrow = ncolumns lol
            print(f" ↪Eval. Loss: {avg_eval_loss:.4f}")
            print(f" ↪FSQ Utilization over the eval. epoch: [{eval_epoch_util}] unique codes.")

    
    torch.save(model.state_dict(), f"./ae_runs/{run_name}/weights/model_{p_count:.1f}m.pt")
    torch.save(decoder.state_dict(), f"./ae_runs/{run_name}/weights/decoder_{dp_count:.2f}m.pt")

    write_config(f"./ae_runs/{run_name}/confs/model_config.json", model_config)
    write_config(f"./ae_runs/{run_name}/confs/decoder_config.json", decoder_config)
    write_config(f"./ae_runs/{run_name}/confs/run_config.json", RUN_PARAMS)
    
    print(f"✅ Training Complete! Weights saved to ae_runs/{run_name}")