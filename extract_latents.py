import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

def extract_and_save_latents(
        model,
        dataloader,
        save_dir,
        chunk_size=10_000,
        device="cuda",
        data="cheetah"
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    latents_buffer = []
    actions_buffer = []
    rewards_buffer = []
    state_buffer = []
    chunk_index = 0
    total_extracted = 0

    dl_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Encoding...")
    with torch.no_grad():
        for idx, (inputs, actions, rewards, *obs_together) in dl_bar:
            inputs = inputs.to(device)

            codes = model(inputs)
            
            latents_buffer.append(codes.cpu())
            actions_buffer.append(actions)
            rewards_buffer.append(rewards)
            if data == "walker":
                obs_orientations, obs_velocities, obs_height = obs_together
                obs_height = obs_height.unsqueeze(-1)
                obs = torch.cat([obs_velocities[:, -1], obs_height[:, -1], obs_orientations[:, -1]], dim=-1)
            else:
                # cheetah here
                obs_position, obs_velocity = obs_together
                obs = torch.cat([obs_position[:, -1], obs_velocity[:, -1]], dim=-1)
            state_buffer.append(obs)

            current_buffer_size = sum(b.size(0) for b in latents_buffer)

            if current_buffer_size >= chunk_size:
                chunk_latents = torch.cat(latents_buffer, dim=0)
                chunk_actions = torch.cat(actions_buffer, dim=0)
                chunk_rewards = torch.cat(rewards_buffer, dim=0)
                chunk_states = torch.cat(state_buffer, dim=0)

                save_path = os.path.join(save_dir, f"latents_chunk_{chunk_index}.pt")
                torch.save({
                    'latents': chunk_latents,
                    'actions': chunk_actions,
                    'rewards': chunk_rewards,
                    'states': chunk_states
                }, save_path)
                print(f"Saved chunk {chunk_index} with {current_buffer_size} samples.")
                
                # Reset buffers
                latents_buffer = []
                actions_buffer = []
                rewards_buffer = []
                state_buffer = []
                chunk_index += 1
                total_extracted += current_buffer_size

    # Save any remaining data in the buffer that didn't perfectly hit the chunk_size
    if latents_buffer:
        chunk_latents = torch.cat(latents_buffer, dim=0)
        chunk_actions = torch.cat(actions_buffer, dim=0)
        chunk_rewards = torch.cat(rewards_buffer, dim=0)
        chunk_states = torch.cat(state_buffer, dim=0)
        
        save_path = os.path.join(save_dir, f"latents_chunk_{chunk_index}.pt")
        
        torch.save({
            'latents': chunk_latents,
            'actions': chunk_actions,
            'rewards': chunk_rewards,
            'states': chunk_states
        }, save_path)
        total_extracted += chunk_latents.size(0)
        
    print(f"Extraction complete. Total samples saved: {total_extracted}")


if __name__ == "__main__":
    from rvsablation.jepa import FSQ_V_JEPA
    from rvsablation.ae import FSQ_VideoVAE
    from rvsablation.utils import seed_everything, read_config, get_dataset
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
        "--sample-num-clips", 
        type=int, 
        default=12, 
        help="Number of clips to sample per trajectory in the dataset."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000,
        help="Extracted data buffer size."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["walker", "cheetah"],
        default="walker"
    )
    args = parser.parse_args()


    rng = seed_everything(seed=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = os.path.join(".", f"{args.model}_runs", f"{args.run_name}")
    save_dir_tr = os.path.join(run_dir, "latents", "training")
    save_dir_ts = os.path.join(run_dir, "latents", "testing")

    dataset, testset, _, _ = get_dataset(rng=rng, 
                                        root=f"./data/{args.dataset}_trajs_h5/state_t4165_f100_r84_s43.h5", 
                                        sample_num_frames=args.sample_num_frames, 
                                        sample_num_clips=args.sample_num_clips, 
                                        data_num_traj=args.data_num_traj, 
                                        data_num_frames=args.data_num_frames,
                                        split_ratio=0.7,
                                        dataset=args.dataset)  
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)

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

    extract_and_save_latents(model, dataloader, save_dir_tr, args.chunk_size, device, args.dataset)
    extract_and_save_latents(model, testloader, save_dir_ts, args.chunk_size, device, args.dataset)
    

