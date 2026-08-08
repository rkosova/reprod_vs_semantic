import argparse

from train_jepa import train as train_j
from train_ae import train as train_a

parser = argparse.ArgumentParser(
    description="Training JEPA or AE."
)

parser.add_argument(
    "--model",
    type=str,
    choices=["jepa", "ae"],
    help="Choice of paradigm to train under."
)

parser.add_argument(
    "--epochs",
    type=int,
    default=30,
    help="Number of training epochs."
)
parser.add_argument(
    "--warmup", 
    type=int, 
    default=10, 
    help="Number of epochs to warm up lr for."
)
parser.add_argument(
    "--lr", 
    type=float, 
    default=2e-4, 
    help="Learning rate after warm-ups."
)
parser.add_argument(
    "--batch", 
    type=int, 
    default=128, 
    help="Mini-batch size."
)
parser.add_argument(
    "--name", 
    type=str, 
    default="fsq__run", 
    help="Run name."
)
parser.add_argument(
    "--seed", 
    type=int, 
    default=42, 
    help="Seed value for randomization."
)
parser.add_argument(
    "--data-num-traj", 
    type=int, 
    default=25_000, 
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

## JEPA Specific
parser.add_argument(
    "--ema-starting-momentum",
    type=float,
    default=0.996,
    help="Starting EMA teacher momentum before annealing."
)
parser.add_argument(
    "--ema-final-momentum", 
    type=float, 
    default=0.999, 
    help="Final EMA teacher momentum after cosine annealing from --ema-starting-momentum."
)
parser.add_argument(
    "--var-reg", 
    type=float, 
    default=0.0, 
    help="Weight for optional variance floor regularization on student context reps."
)
# Best to compute this directly from the levels, making an assumption of Gaussian or
# Uniform distribution post-training
# if we try to do it here it will hardcode the fsq dim size.
# parser.add_argument(
#     "--var-target-std", 
#     type=float, 
#     default=0.1, 
#     help="Target per-feature std for --var-reg."
# ) 
### Masking
parser.add_argument(
    "--short-range-targets",
    type=int,
    default=2,
    help="Number of short range target blocks to be sampled."
)
parser.add_argument(
    "--long-range-targets",
    type=int,
    default=2,
    help="Number of long range target blocks to be sampled."
)
parser.add_argument(
    "--short-scale",
    nargs=2,
    type=float,
    default=(.1, .15),
    help="Relative scale range of the sampled short range targets."
)
parser.add_argument(
    "--long-scale",
    nargs=2,
    type=float,
    default=(.2, .35),
    help="Relative scale range of the sampled long range targets."
)
parser.add_argument(
    "--context-scale",
    nargs=2,
    type=float,
    default=(1., 1.),
    help="Relative scale range of the sampled context."
)
parser.add_argument(
    "--short-ar",
    nargs=2,
    type=float,
    default=(.75, 1.5),
    help="Aspect ratio range of the sampled short range targets."
)
parser.add_argument(
    "--long-ar",
    nargs=2,
    type=float,
    default=(.35, 1.5),
    help="Aspect ratio range of the sampled long range targets."
)
parser.add_argument(
    "--context-ar",
    nargs=2,
    type=float,
    default=(1., 1.),
    help="Aspect ratio range of the sampled context."
)



args = parser.parse_args()

if args.model == "jepa":
    train_j(
        epochs=args.epochs,
        warmup_epochs=args.warmup,
        batch_size=args.batch,
        lr=args.lr,
        run_name=args.name,
        seed=args.seed,
        data_num_traj=args.data_num_traj,
        data_num_frames=args.data_num_frames,
        sample_num_frames=args.sample_num_frames,
        sample_num_clips=args.sample_num_clips,
        ema_starting_momentum=args.ema_starting_momentum,
        ema_final_momentum=args.ema_final_momentum,
        var_reg=args.var_reg,
        short_range_targets=args.short_range_targets,
        long_range_targets=args.long_range_targets,
        short_scale=args.short_scale,
        short_ar=args.short_ar,
        long_scale=args.long_scale,
        long_ar=args.long_ar,
        context_scale=args.context_scale,
        context_ar=args.context_ar
    )
elif args.model == "ae":
    train_a(
        epochs=args.epochs,
        warmup_epochs=args.warmup,
        batch_size=args.batch,
        lr=args.lr,
        run_name=args.name,
        seed=args.seed,
        data_num_traj=args.data_num_traj,
        data_num_frames=args.data_num_frames,
        sample_num_frames=args.sample_num_frames,
        sample_num_clips=args.sample_num_clips,
    )