
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics.regression import R2Score

from rvsablation.core import _TransformerStack
from rvsablation.utils import get_2d_sincos_pos_embed

import math

import glob
import os

class PrecomputedLatentDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()

        self.chunk_files = sorted(
            glob.glob(
                os.path.join(data_dir, "latents_chunk_*.pt")
            )
        )

        if not self.chunk_files:
            raise FileNotFoundError(f"No latent chunks found in {data_dir}!")

        all_latents = []
        all_actions = []
        all_rewards = []
        all_states =  []

        for f in self.chunk_files:
            data = torch.load(f, weights_only=True)
            all_latents.append(data["latents"])
            all_actions.append(data['actions'])
            all_rewards.append(data['rewards'])
            all_states.append(data['states'])

        self.latents = torch.cat(all_latents, dim=0)
        self.actions = torch.cat(all_actions, dim=0)
        self.rewards = torch.cat(all_rewards, dim=0)
        self.states = torch.cat(all_states, dim=0)

    def __len__(self):
        return self.latents.size(0)

    def __getitem__(self, index):
        return {
            'latents': self.latents[index],
            'actions': self.actions[index],
            'rewards': self.rewards[index],
            'states': self.states[index]
        }


class PrecompLatentTransitionDataset(Dataset):
    def __init__(self, 
                 data_dir,
                 num_tubelets):
        super().__init__()
        self.parent = PrecomputedLatentDataset(data_dir)
        self.num_tubelets = num_tubelets
        self.tubelet_transitions = num_tubelets - 1

        self.frames_per_tubelet = self.parent[0]["actions"].shape[0] // num_tubelets
        self.tokens_per_tubelet = self.parent[0]["latents"].shape[0] // num_tubelets

    def __len__(self):
        return len(self.parent) * self.tubelet_transitions

    def __getitem__(self, index):
        super_index = index // self.tubelet_transitions
        sub_index = index % self.tubelet_transitions

        super_sample = self.parent[super_index]
        super_latents = super_sample["latents"].reshape(self.num_tubelets,
                                                        self.tokens_per_tubelet,
                                                        -1)
        super_actions = super_sample["actions"]

        present = super_latents[sub_index]
        action = super_actions[(sub_index + 1) * self.frames_per_tubelet - 1]
        future = super_latents[sub_index + 1]

        return present, action, future, super_latents, super_actions


class TransitionProbe(nn.Module):
    def __init__(self,
                 spatial_grid_size,
                 latent_dim,
                 action_dim,
                 hidden_dim,
                 depth,
                 attention_dim,
                 attention_num_heads, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_grid_size = spatial_grid_size
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim, bias=False)
        self.action_proj = nn.Linear(self.action_dim, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.latent_dim, bias=False)

        self.stack = _TransformerStack(hidden_dim, depth, attention_dim, attention_num_heads)

        pos = get_2d_sincos_pos_embed(hidden_dim, grid_size=spatial_grid_size)
        self.register_buffer(
            "pos",
            pos.unsqueeze(0),
            persistent=False
        )

    def forward(self, x, a):
        x = self.latent_proj(x)
        a = self.action_proj(a).unsqueeze(1)

        x = x + a + self.pos # additive action fusion + positional embedding

        out = self.stack(x, attn_mask=None)

        return self.out_proj(out)


def get_probe_architecture(
        probe_type,
        latent_dim,
        out_dim=None,
        hidden_dim=None,
        action_dim=None,
        num_tokens=None,
        depth=None,
        attention_dim=None,
        attention_num_heads=None
):
    if probe_type == 'linear':
        assert out_dim != None
        return nn.Linear(latent_dim, out_dim)
    
    elif probe_type == 'mlp':
        assert out_dim != None
        return nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    elif probe_type == 'transition':
        assert num_tokens and hidden_dim and action_dim and depth and attention_dim and attention_num_heads
        spatial_grid_size = math.isqrt(num_tokens)
        assert spatial_grid_size ** 2 == num_tokens, "`num_tokens` must have integer square root"
        
        return TransitionProbe(
            spatial_grid_size=spatial_grid_size,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            attention_dim=attention_dim,
            attention_num_heads=attention_num_heads
        )
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")

    
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        latents = batch['latents'].to(device)
        latents = latents.reshape(latents.shape[0], -1)
        targets = batch['states'].to(device)
        
        optimizer.zero_grad()
        preds = model(latents)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def evaluate_epoch(model, dataloader, criterion, r2_metric, device):
    model.eval()
    total_loss = 0.0
    r2_metric.reset()
    
    with torch.no_grad():
        for batch in dataloader:
            latents = batch['latents'].to(device)
            latents = latents.reshape(latents.shape[0], -1)
            targets = batch['states'].to(device)
            
            preds = model(latents)
            loss = criterion(preds, targets)
            total_loss += loss.item()
            
            r2_metric.update(preds, targets)
            
    avg_loss = total_loss / len(dataloader)
    
    r2_per_dim = r2_metric.compute() 
    mean_r2 = r2_per_dim.mean().item()
    
    return avg_loss, mean_r2, r2_per_dim.cpu().numpy()


if __name__ == "__main__":
    import argparse

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="Train Downstream Probes")
    parser.add_argument('--probe-type', type=str, required=True, choices=['linear-state', 'mlp-state', 'transition-transition'])
    parser.add_argument('--model-type', type=str,required=True, choices=["jepa", "ae"])
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--num-tubelets', type=int, default=4)
    args = parser.parse_args()

    probe_type, task_type = args.probe_type.split("-")
    data_dir = os.path.join(".", f"{args.model_type}_runs", args.run_name, "latents")
    print(f"📁 Loading latents from {data_dir}")

    if task_type == "state" or task_type == "task":
        train_dataset = PrecomputedLatentDataset(os.path.join(data_dir, "training"))
        test_dataset = PrecomputedLatentDataset(os.path.join(data_dir, "testing"))

        if task_type == "state":
            task_dim = train_dataset[0]["states"].shape[0]
            criterion = nn.MSELoss()
            metric = R2Score(multioutput="raw_values").to(device)
        # elif task_type == "task" then criterion = crossentropy
    elif task_type == "transition":
        train_dataset = PrecompLatentTransitionDataset(os.path.join(data_dir, "training"), num_tubelets=args.num_tubelets)
        test_dataset = PrecompLatentTransitionDataset(os.path.join(data_dir, "testing"), num_tubelets=args.num_tubelets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    N, D = train_dataset[0]["latents"].shape
    # for now, we only consider flattening tokens to a joint representation
    probe = get_probe_architecture(probe_type, latent_dim=N * D, out_dim=task_dim).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=args.lr)

    print(f"--- Training {args.probe_type.upper()} Probe ---")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(probe, train_loader, optimizer, criterion, device)
        val_loss, val_r2, r2_breakdown = evaluate_epoch(probe, test_loader, criterion, metric, device)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val Mean R2: {val_r2:.4f}")

    print("\n--- Final Dimensional R2 Breakdown ---")
    # This matches up with the 24 dimensions of the Walker state
    for i, r2_val in enumerate(r2_breakdown):
        print(f"Dim {i:02d}: {r2_val:.4f}")
    print(f"Mean R^2: {val_r2}")