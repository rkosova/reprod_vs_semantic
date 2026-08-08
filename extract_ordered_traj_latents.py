"""Extract adjacent sliding-window latent transitions from full trajectories.

Each encoded state is an entire ``num_frames_clip``-frame window.  For example,
with 8-frame clips and 2-frame tubelets, adjacent states are:

    E(frames[0:8]) -> E(frames[2:10])

The target is encoded independently, so overlapping tubelets retain their
new, shifted-window representations.
"""

import os

import torch
from tqdm import tqdm


def extract_and_save_traj_latents(
    num_frames_clip,
    num_frames_tubelet,
    model,
    dataloader,
    save_dir,
    chunk_size=10_000,
    device="cuda",
):
    """Encode and save sliding-window transition examples.

    The saved chunk dictionaries contain:

    - ``latents``: present window states, shape ``(examples, tokens, dim)``.
    - ``future_latents``: independently encoded next window states.
    - ``actions``: departing action of the final tubelet in each present state.
    - ``rewards`` and ``states``: metadata from the end of each future window.

    Dataset items must provide full trajectories as
    ``(frames, actions, rewards, orientations, velocities, heights)``.  The
    trajectory axis may have any length at least ``num_frames_clip``.
    """
    if num_frames_clip % num_frames_tubelet != 0:
        raise ValueError("num_frames_clip must be divisible by num_frames_tubelet")

    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    model.eval()

    latents_buffer = []
    future_latents_buffer = []
    actions_buffer = []
    rewards_buffer = []
    states_buffer = []
    chunk_index = 0
    total_extracted = 0

    def save_chunk():
        """Save and reset the current CPU buffers."""
        nonlocal chunk_index, total_extracted

        chunk_latents = torch.cat(latents_buffer, dim=0)
        chunk_future_latents = torch.cat(future_latents_buffer, dim=0)
        chunk_actions = torch.cat(actions_buffer, dim=0)
        chunk_rewards = torch.cat(rewards_buffer, dim=0)
        chunk_states = torch.cat(states_buffer, dim=0)

        save_path = os.path.join(save_dir, f"latents_chunk_{chunk_index}.pt")
        torch.save(
            {
                "latents": chunk_latents,
                "future_latents": chunk_future_latents,
                "actions": chunk_actions,
                "rewards": chunk_rewards,
                "states": chunk_states,
            },
            save_path,
        )
        num_examples = chunk_latents.size(0)
        print(f"Saved chunk {chunk_index} with {num_examples} transitions.")

        latents_buffer.clear()
        future_latents_buffer.clear()
        actions_buffer.clear()
        rewards_buffer.clear()
        states_buffer.clear()
        chunk_index += 1
        total_extracted += num_examples

    dl_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Encoding")
    with torch.no_grad():
        for _, batch in dl_bar:
            frames, actions, rewards, orientations, velocities, heights = batch
            batch_size, trajectory_frames = frames.shape[:2]

            window_index_cpu = torch.arange(trajectory_frames).unfold(
                dimension=0,
                size=num_frames_clip,
                step=num_frames_tubelet,
            )
            num_windows = window_index_cpu.size(0)
            if num_windows < 2:
                raise ValueError("A trajectory must contain at least two sliding windows")

            # Shape: (batch, windows, frames_per_window, channels, height, width).
            windowed_frames = frames[:, window_index_cpu]
            _, _, clip_frames, channels, height, width = windowed_frames.shape
            encoder_input = windowed_frames.to(device).reshape(
                batch_size * num_windows,
                clip_frames,
                channels,
                height,
                width,
            )

            # Shape: (batch, windows, tokens, latent_dim).
            encoded_flat = model(encoder_input)
            encoded_windows = encoded_flat.reshape(
                batch_size,
                num_windows,
                *encoded_flat.shape[1:],
            )

            # Adjacent encoded windows form the window-state transition data.
            present = encoded_windows[:, :-1].flatten(0, 1).cpu()
            future = encoded_windows[:, 1:].flatten(0, 1).cpu()

            # Departing action from present to future
            windowed_actions = actions[:, window_index_cpu]
            transition_actions = windowed_actions[:, :-1, -1].flatten(0, 1).cpu() # actions are first aligned with present windows and the last action of each present window is taken

            # Keep metadata aligned with the target (future) state.
            windowed_rewards = rewards[:, window_index_cpu]
            transition_rewards = windowed_rewards[:, 1:, -1].flatten(0, 1).cpu()

            windowed_orientations = orientations[:, window_index_cpu]
            windowed_velocities = velocities[:, window_index_cpu]
            windowed_heights = heights[:, window_index_cpu]
            transition_states = torch.cat(
                (
                    windowed_velocities[:, 1:, -1],
                    windowed_heights[:, 1:, -1].unsqueeze(-1),
                    windowed_orientations[:, 1:, -1],
                ),
                dim=-1,
            ).flatten(0, 1).cpu()

            latents_buffer.append(present)
            future_latents_buffer.append(future)
            actions_buffer.append(transition_actions)
            rewards_buffer.append(transition_rewards)
            states_buffer.append(transition_states)

            current_buffer_size = sum(tensor.size(0) for tensor in latents_buffer)
            if current_buffer_size >= chunk_size:
                save_chunk()

    if latents_buffer:
        save_chunk()

    print(f"Extraction complete. Total transitions saved: {total_extracted}")


if __name__ == "__main__":
    from rvsablation.jepa import FSQ_V_JEPA
    from rvsablation.ae import FSQ_VideoVAE
    from rvsablation.utils import seed_everything, read_config, get_dataset
    from torch.utils.data import DataLoader
    import os
    # import sys
    from pathlib import Path
    # parent_dir = str(Path('..').resolve())
    # if parent_dir not in sys.path:
    #     sys.path.append(parent_dir)
    # from dmc_dataset.dataset import get_dataset

    import argparse

    parser = argparse.ArgumentParser(
        description="Extract latents from pretrained JEPA or AE."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["jepa", "ae"],
        help="Choosing which pretrained model to use."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name of training run to use the model from."
    )
    parser.add_argument(
        "--data-num-traj", 
        type=int, 
        default=4165, 
        help="Number of trajectories in generated dataset. (Can be less than true number.)"
    )
    parser.add_argument(
        "--data-num-frames", 
        type=int, 
        default=100, 
        help="Number of frames per dataset trajectory."
    )
    parser.add_argument(
        "--sample-num-frames", 
        type=int, 
        default=8, 
        help="Number of frames per clip sampled."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000,
        help="Extracted data buffer size."
    )
    parser.add_argument(
        "--num-frames-tubelet",
        type=int,
        default=2
    )
    args = parser.parse_args()


    rng = seed_everything(seed=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = os.path.join(".", f"{args.model}_runs", f"{args.run_name}")
    save_dir_tr = os.path.join(run_dir, "latents_transition", "training")
    save_dir_ts = os.path.join(run_dir, "latents_transition", "testing")

    dataset, testset, _, _ = get_dataset(rng=rng, 
                                        root="./data/walker_trajs_h5/state_t4165_f100_r84_s43.h5", 
                                        sample_num_frames=args.data_num_frames, 
                                        sample_num_clips=1, 
                                        data_num_traj=args.data_num_traj, 
                                        data_num_frames=args.data_num_frames,
                                        split_ratio=0.7)  
    dataloader = DataLoader(dataset, batch_size=25, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=25, shuffle=True, num_workers=4, pin_memory=True)

    conf_path = os.path.join(run_dir, "confs", "model_config.json")
    print(f"Loading model config from {conf_path}")
    weight_files = list((Path(run_dir) / "weights").glob("model_*.pt"))
    weights_path = weight_files[0]
    print(f"Loading model weights from {weights_path}")
    model_config = read_config(conf_path)

    model = FSQ_V_JEPA(**model_config) if args.model == "jepa" else FSQ_VideoVAE(**model_config) if args.model == "ae" else None
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )


    model.to(device)

    extract_and_save_traj_latents(args.sample_num_frames, args.num_frames_tubelet, model, dataloader, save_dir_tr, args.chunk_size, device)
    extract_and_save_traj_latents(args.sample_num_frames, args.num_frames_tubelet, model, testloader, save_dir_ts, args.chunk_size, device)