import torch
from torch.utils.data import Dataset
import random
import numpy as np

import os
import json
import glob
import h5py



def patchify_clip(x: torch.Tensor, tubelet_size: int, patch_size: int):
    """
    :param x: Input clip of tensor shape (B, T, C, H, W)
    :param tubelet_size: Number of frames in a tubelet, must divide duration T of clip
    :param patch_size: Size of spatial patch side in pixels, must divide H and W.
    """
    B, T, C, H, W = x.shape
    assert T % tubelet_size == 0, "Duration of input is not perfectly divisible by tubelete_size"
    assert H % patch_size == 0, "Height of input frames is not perfectly divisible by patch_size"
    assert W % patch_size == 0, "Width of input frames is not perfectly divisible by patch_size"

    Ts = T//tubelet_size
    Ps = H // patch_size

    x = x.reshape(
        B,
        Ts,
        tubelet_size,
        C,
        Ps,
        patch_size,
        Ps,
        patch_size
    ) # (B, T, C, H, W) --> (B, T/t, t, C, H/p, p, W/p, p)
    x = x.permute(0, 1, 4, 6, 2, 3, 5, 7) # (B, T/t, t, C, H/p, p, W/p, p) --> (B, T/t, H/p, W/p, t, C, p, p)
    x = x.reshape(B, Ts * Ps * Ps, tubelet_size * C * patch_size * patch_size) # (B, T/t, H/p, W/p, t, C, p, p) --> (B, N = T/t H/p W/p, D = t C p p)

    return x


def unpatchify_clip(x: torch.Tensor, duration: int, tubelet_size: int, image_size: int, patch_size: int):
    """
    :param x: Patched input of shape (B, N, D) where N = T/t H/p W/p and D = t C p p where T = duration, H, W = image_size, t = tubelet_size, p = patch_size
    :param duration: Number of frames in clip.
    :param tubelet_size: Number of frames in a tubelet.
    :param image_size: Size of image side in pixels (assuming square)
    :param patch_size: Size of patch side in pixels
    """
    B, _, D = x.shape
    Ts = duration // tubelet_size
    Ps = image_size // patch_size
    C = D // (tubelet_size * patch_size * patch_size)
    
    x = x.reshape(B, Ts, Ps, Ps, tubelet_size, C, patch_size, patch_size)
    x = x.permute(0, 1, 4, 5, 2, 6, 3, 7) # (B, T/t, H/p, W/p, t, C, p, p) --> (B, T/t, t, C, H/p, p, W/p, p)
    x = x.reshape(B, duration, C, image_size, image_size) # (B, T/t, t, C, H/p, p, W/p, p) --> (B, T, C, H, W)

    return x


def get_1d_sincos_pos_embed(dim, positions):
    assert dim % 2 == 0, "Dimension of embeddings must be divisible by 2."

    omega = torch.arange(dim // 2, dtype=torch.float32)
    omega = 1. / (10000 ** (omega / (dim / 2)))

    out = positions[:, None] * omega[None, :]

    embed_sin = torch.sin(out)
    embed_cos = torch.cos(out)

    return torch.cat([embed_sin, embed_cos], dim=1)


def get_2d_sincos_pos_embed(dim, grid_size):
    assert dim % 4 == 0

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)

    grid = torch.meshgrid(
        grid_h,
        grid_w,
        indexing="ij"
    )

    grid = torch.stack(grid, dim=0)
    grid = grid.reshape(2, -1)

    emb_h = get_1d_sincos_pos_embed(
        dim // 2,
        grid[0]
    )

    emb_w = get_1d_sincos_pos_embed(
        dim // 2,
        grid[1]
    )

    return torch.cat([emb_h, emb_w], dim=1)

def get_3d_sincos_pos_embed(dim, temporal_grid_size, grid_size):
    assert dim % 6 == 0, "Dimension of embeddings must be divisible by 6."

    grid_t = torch.arange(temporal_grid_size, dtype=torch.float32)
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)

    grid = torch.meshgrid(
        grid_t,
        grid_h,
        grid_w,
        indexing="ij"
    )

    grid = torch.stack(grid)
    grid = grid.reshape(3, -1)

    emb_t = get_1d_sincos_pos_embed(
        dim // 3,
        grid[0]
    )
    emb_h = get_1d_sincos_pos_embed(
        dim // 3,
        grid[1]
    )
    emb_w = get_1d_sincos_pos_embed(
        dim // 3,
        grid[2]
    )

    return torch.cat([emb_t, emb_h, emb_w], dim=1)


def sample_block(spatial_grid_size, scale_range, ar_range) -> tuple[int, int, int, int]:
    """
    Samples a random block within a grid of spatial_grid_size^2 in the given range and aspect ratio.

    Tries 10 times to sample a block within the given ranges that respects the size of the grid. If it fails, it will return a block
    half the scale of the grid positioned at the top left corner.

    :param spatial_grid_size: The number of spatial positions per grid side.
    :param scale_range: Range of possible relative scales of the sampled block. Scale is `area / spatial_grid_size^2`.
    :param ar_range: Range of possible aspect ratios of the sampled block. Aspect ratio is `height / width`.
    :return top, left, h, w: Tuple of 4 integers (top, left, height, width) defining the top-left corner of the block and its width and height.
    """
    for _ in range(10):
        s = torch.empty(1).uniform_(*scale_range).item() * (spatial_grid_size ** 2) # patch area: s = hw
        ar = torch.empty(1).uniform_(*ar_range).item() # aspect ratio: ar = h / w
        h = round((s * ar) ** .5) # h = sqrt(h^2) = sqrt(h / w * hw)
        w = round((s / ar) ** .5) # ar = sqrt(w^2) = sqrt(hw / (h / w))
        if 0 < h <= spatial_grid_size and 0 < w <= spatial_grid_size:
            top = torch.randint(0, spatial_grid_size - h + 1, (1,)).item()
            left = torch.randint(0, spatial_grid_size - w + 1, (1,)).item()
            return top, left, h, w

    h = w = max(1, spatial_grid_size // 2) 
    return 0, 0, h, w # top left half size block as default


def block_to_flat_idx(t_start, top, left, d, h, w, spatial_grid_size) -> torch.Tensor:
    """
    Given a block defined by `(top, left, h, w)` it expands the block through time, starting at `t_start` through to `t_start + d - 1`, and it returns a tensor
    of shape `(K,)`, where `K` is the number of spatio-temporal positions the block encompasses. This tensor represents a temporal-major flattening of the indices
    of the block relative to the 3D grid (time x height x width).

    :param t_start: Starting temporal position.
    :param top: Spatial row where block starts.
    :param left: Spatial column where block starts.
    :param d: Temporal duration of block.
    :param h: Height of block.
    :param w: Width of block.
    :param spatial_grid_size: The number of spatial positions per grid side.

    :return flat_indices: Temporal-major flattening of the given block.
    """
    ts = torch.arange(t_start, t_start + d)
    rows = torch.arange(top, top + h)
    cols = torch.arange(left, left+w)
    tt, rr, cc = torch.meshgrid(ts, rows, cols, indexing="ij") # (d, h, w) where tt[i, j, k] = ts[i], rr[i, j, k] = rows[j], and cc[i, j, k] = cols[k]
    # if we think of (d, h, w) as the tube representation of the block:
    #       - tt[i, j, k] gives us the absolute temporal position of the tubelet at the i-th index of our temporal range ts, j-th index of our row range rows, and k-th index of our column range cols
    #       - rr[i, j, k] gives us the absolute row position ...
    #       - cc[i, j, k] gives us the abolute column position ... 
    return (tt * (spatial_grid_size ** 2) + rr * spatial_grid_size + cc).flatten() # temporal-then-row-then-column flattening. one temporal step encompasses G^2 tubelets and one row G tubelets.


def generate_multi_block_mask(spatial_grid_size, temporal_grid_size, short_range_targets=2, long_range_targets=2,
                              short_scale=(.1, .15), short_ar=(.75, 1.5), long_scale=(.2, .35), long_ar=(.35, 1.5),
                              context_scale=(1., 1.), context_ar=(1., 1.)) -> tuple[torch.Tensor, list]:
    """
    Generates the number of the given masks with the given scale and aspect ratio ranges. Also samples context and removes overlap with targets.
    Under default context parameters, it takes the difference of the entire image with the target union as the context.

    :return ctx_idx: Tensor of shape `(Kc, )` of flattened sorted context indices, where `Kc` is the number of positions in the 3D grid encompassed by the context.
    :return target_idx_list: List of length `short_range_targets + long_range_targets` containing tensors of different sizes. Each tensor is the flattened
    index tensor of a sampled block. The blocks may overlap.
    """
    N = temporal_grid_size * (spatial_grid_size ** 2)
    target_idx_list = []
    target_mask = torch.zeros(N, dtype=torch.bool)

    sample_configs = [
        (short_range_targets, short_scale, short_ar),
        (long_range_targets, long_scale, long_ar)
    ]

    for num_targets, scale, ar in sample_configs:
        for _ in range(num_targets):
            top, left, h, w = sample_block(spatial_grid_size, scale, ar)
            idx = block_to_flat_idx(0, top, left, temporal_grid_size, h, w, spatial_grid_size)
            target_idx_list.append(idx)
            target_mask[idx] = True

    ctx_mask = torch.zeros(N, dtype=torch.bool)
    top, left, h, w = sample_block(spatial_grid_size, context_scale, context_ar)
    ctx_mask[block_to_flat_idx(0, top, left, temporal_grid_size, h, w, spatial_grid_size)] = True
    ctx_mask &= ~target_mask
    # original patchification is time-major, meaning all positions in the 3D grid in the same time step are contiguous. 
    # let G = spatial_grid_size and T = temporal_grid_size
    # t = 1: 0            , 1                , ..., G^2 - 1
    # t = 2: G^2          , G^2 + 1          , ..., 2 * G^2 - 1
    # t = T, (T - 1) * G^2, (T - 1) * G^2 + 1, ..., T * G^2 - 1
    # nonzero sorts the context indices to this same order (so ctx_idx is a sub-ordered-set of the full indexing scheme)
    # this is not necessarily required for e.g. attention as the input is gathered by jepa_mask in the order of ctx_idx
    # and the causal attention mask is similarly indexed by ctx_idx along both dimensions 
    ctx_idx = ctx_mask.nonzero(as_tuple=True)[0] 
    return ctx_idx, target_idx_list


def jepa_mask(x, indices):
    B, _, D = x.shape
    return torch.gather(x, dim=1, index=indices.view(1, -1, 1).expand(B, -1, D).to(x.device))


def build_predictor_input(ctx, ctx_idx, tgt_idx_list, pos_embed, mask_token) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Builds input for predictor by concatinating mask tokens to end of context input and adding positional embeddings based on provided context and target indices.

    :param ctx: Context tokens. Shape: (B, Kc, D)
    :param ctx_idx:  Tensor of flattened context indices. Shape: `(Kc, )` where `Kc` is the number of positions in the 3D grid encompassed by the context.
    :param tgt_idx_list: List of flattened target indices. One tensor per target.
    :param pos_embed: Positional embedding to be added to the context and target positions. Shape: `(N, D, )` where `N` is the total number of positions in the 
    3D grid and `D` is the dimension of the context.
    :param mask_token: Mask token to represent target positions. Shape: `(D, )` where `D` is the dimension of the context.

    :return full_in: Tensor containing context and mask tokens with appropriate positional information. Shape: `(B, N, D)` where `N = Kc + Kt` such that the first `Kc` elements represent the context positions.
    :return full_idx: Tensor containing the indices of the context and target positions in 3D grid. Shape: `(B, N, )` where `N = Kc + Kt` such that the first `Kc` elements represent the context indices. One-to-one with `full_in`.
    :return Kc: Number of context positions in `full_in` per sample.
    """
    B, Kc, _ = ctx.shape
    ctx_pos = pos_embed[ctx_idx].expand(B, -1, -1)
    ctx_in = ctx + ctx_pos

    all_target_idx = torch.unique(torch.cat(tgt_idx_list), sorted=True, dim=None) # returns unique elements of the flattened tgt_idx_list in sorted (ascending) order
    Kt = all_target_idx.shape[0]                                                  # therefore its the union of all target indices sampled earlier, in sorted order
    tgt_pos = pos_embed[all_target_idx].expand(B, -1, -1)
    mask_token = mask_token.expand(B, Kt, -1)
    tgt_in = mask_token + tgt_pos

    full_idx = torch.cat([ctx_idx, all_target_idx]) # (Kc + Kt, )
    full_in = torch.cat([ctx_in, tgt_in], dim=1) # (B, Kc + Kt, D)

    return full_in, full_idx, Kc

def seed_everything(seed=42):
    """
    Locks down all random number generators for full reproducibility.
    """
    # 1. Python built-in random
    random.seed(seed)
    
    # 2. Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. NumPy
    rng = np.random.default_rng(seed)
    
    # 4. PyTorch CPU
    torch.manual_seed(seed)
    
    # 5. PyTorch GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # If you use multi-GPU
    
    # 6. CuDNN Backend (Forces CUDA to use deterministic algorithms)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🌱 Global seed set to {seed}")

    return rng


def read_config(path):
    with open(path) as f:
        conf = json.load(f)

    if "fsq_levels" in conf:
        fsq_levels = conf["fsq_levels"]
        
        assert type(fsq_levels) == list, "Parameter `fsq_levels` must be a list."
        assert len(fsq_levels) == conf["fsq_dim"], "Length of `fsq_levels` must match the `fsq_dim` value."

        fsq_levels = torch.Tensor(fsq_levels).reshape(1, 1, -1)
    
        conf["fsq_levels"] = fsq_levels
    
    return conf

def write_config(path, config):
    with open(path, "w") as f:
        if "fsq_levels" in config:
            config["fsq_levels"] = config["fsq_levels"].reshape(-1).tolist()
        json.dump(config, f)

@torch.no_grad()
def fsq_2_index(q, radii, levels_max):
    """
    Takes quantized using levels_max // 2 as levels and computes a unique integer index.
    
    :param q: FSQ quantized vectors. (N, D)
    :param radii: Per-dimension-radix. (1, D)
    :param levels_max: Maximum value per dimension. (1, D)
    """
    return ((q + levels_max) * radii).sum(dim=1).to(torch.long)


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
        all_future_latents = []
        all_actions = []
        all_rewards = []
        all_states =  []

        for f in self.chunk_files:
            data = torch.load(f, weights_only=True)
            all_latents.append(data["latents"])
            if "future_latents" not in data:
                raise KeyError(
                    f"{f} does not contain 'future_latents'. "
                    "Use window-transition chunks produced by "
                    "extract_ordered_traj_latents.py."
                )
            all_future_latents.append(data["future_latents"])
            all_actions.append(data['actions'])
            all_rewards.append(data['rewards'])
            all_states.append(data['states'])

        self.latents = torch.cat(all_latents, dim=0)
        self.future_latents = torch.cat(all_future_latents, dim=0)
        self.actions = torch.cat(all_actions, dim=0)
        self.rewards = torch.cat(all_rewards, dim=0)
        self.states = torch.cat(all_states, dim=0)

    def __len__(self):
        return self.latents.size(0)

    def __getitem__(self, index):
        return {
            'latents': self.latents[index],
            'future_latents': self.future_latents[index],
            'actions': self.actions[index],
            'rewards': self.rewards[index],
            'states': self.states[index]
        }

def get_dataset(rng: np.random.Generator,
                root: str,
                sample_num_frames: int=4,
                sample_num_clips: int=15,
                data_num_traj: int=100, 
                data_num_frames: int=100,
                stride: int=1,
                split_ratio: float=0.9,
                train_transform=None): # <-- Add transform argument
    
    traj_ids = np.arange(data_num_traj)
    indices = generate_trajectory_sample_idx(rng, sample_num_frames, sample_num_clips, data_num_traj, data_num_frames, stride)

    split_idx = int(data_num_traj * split_ratio)

    train_indices = indices[:split_idx, :] 
    test_indices = indices[split_idx:, :] 
    train_traj_ids = traj_ids[:split_idx]
    test_traj_ids = traj_ids[split_idx:]

    # Apply the transform ONLY to the training set. 
    # The test set should remain deterministic for clean evaluation.
    train_ds = WalkerHDF5(root, train_indices, train_traj_ids, stride, sample_num_frames, transform=train_transform)
    test_ds = WalkerHDF5(root, test_indices, test_traj_ids, stride, sample_num_frames, transform=None)

    return train_ds, test_ds, train_indices, test_indices

def generate_trajectory_sample_idx(rng: np.random.Generator, 
                                   sample_num_frames: int=4,
                                   sample_num_clips: int=15,
                                   data_num_traj: int=100, 
                                   data_num_frames: int=100,
                                   stride: int=1):
    """Return valid clip start indices for every trajectory.

    A window of ``sample_num_frames`` frames with stride ``stride`` occupies
    ``(sample_num_frames - 1) * stride + 1`` source frames.  Starts are
    inclusive, so a window that exactly fills a trajectory still has the one
    valid start index, zero.
    """
    # span({i + 0*s, i + 1*s, ..., i + (l - 1)*s}) = |{i, i+1, i+2, ..., i + (l-1)s}| = i + (l - 1)s - i + 1 = (l-1)s + 1  
    span_of_sampled_window = (sample_num_frames - 1) * stride + 1 
    num_valid_starts = data_num_frames - span_of_sampled_window + 1

    if num_valid_starts <= 0:
        raise ValueError(
            "The requested clip does not fit in a trajectory: "
            f"sample_num_frames={sample_num_frames}, stride={stride}, "
            f"data_num_frames={data_num_frames}."
        )
    if sample_num_clips > num_valid_starts:
        raise ValueError(
            f"Requested {sample_num_clips} clips per trajectory, but only "
            f"{num_valid_starts} non-overlapping start positions are available "
            "when sampling without replacement."
        )

    return rng.permuted(
        np.tile(np.arange(num_valid_starts), (data_num_traj, 1)),
        axis=1
    )[:, :sample_num_clips]


class WalkerHDF5(Dataset):
    def __init__(self,
                 root: str,
                 sample_indices: np.ndarray,
                 sample_traj_ids: np.ndarray,
                 stride: int=1,
                 sample_num_frames: int=4,
                 transform=None): # <-- Added transform
        super().__init__()
        self.root = root
        self.sample_indices = sample_indices
        self.sample_traj_ids = sample_traj_ids
        self.stride = stride
        self.sample_num_frames = sample_num_frames
        self.transform = transform # <-- Store it

        self.num_traj, self.frames_per_traj = self.sample_indices.shape
        self.total_samples = self.sample_indices.size
        self.db = None

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, index):
        if self.db is None:
            self.db = h5py.File(self.root, 'r')

        local_i = index // self.frames_per_traj 
        j = index % self.frames_per_traj

        traj_i = self.sample_traj_ids[local_i] 

        start = self.sample_indices[local_i, j]
        end = start + self.stride * self.sample_num_frames

        frames = self.db['frames'][traj_i, start:end:self.stride]
        actions = self.db['actions'][traj_i, start:end:self.stride]
        rewards = self.db['rewards'][traj_i, start:end:self.stride]
        obs_orientations = self.db['obs_orientations'][traj_i, start:end:self.stride]
        obs_velocity = self.db['obs_velocity'][traj_i, start:end:self.stride]
        obs_height = self.db['obs_height'][traj_i, start:end:self.stride]

        frames_tensor = torch.from_numpy(frames).float() / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2) # (T, C, H, W)
        
        # --- NEW: Apply transform if it exists ---
        if self.transform is not None:
            frames_tensor = self.transform(frames_tensor)

        actions_tensor = torch.from_numpy(actions)
        rewards_tensor = torch.from_numpy(rewards)
        obs_orientations_tensor = torch.from_numpy(obs_orientations)
        obs_velocity_tensor = torch.from_numpy(obs_velocity)
        
        return frames_tensor, actions_tensor, rewards_tensor, \
                obs_orientations_tensor, obs_velocity_tensor, obs_height
