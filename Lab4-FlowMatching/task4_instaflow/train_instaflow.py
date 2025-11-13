"""
Training Script for InstaFlow One-Step Generator (Task 4 - Phase 2)

Trains a one-step generator via distillation from a 2-Rectified Flow teacher model.
The student model learns to generate high-quality images in a single forward pass
by mimicking the teacher's multi-step outputs.

Reference: Liu et al., "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation"

Usage:
    python -m task4_instaflow.train_instaflow \
        --distill_data_path data/afhq_instaflow \
        --use_cfg \
        --train_num_steps 100000 \
        --warmup_steps 5000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
# Import schedulers needed for Linear Warmup + Cosine Decay
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)
from dotmap import DotMap
from pytorch_lightning import seed_everything
from tqdm import tqdm

sys.path.append('.')
from image_common.dataset import tensor_to_pil_image
from image_common.instaflow import InstaFlowModel
from image_common.network import UNet

from task4_instaflow.instaflow_dataset import (
    InstaFlowDataset,
    get_instaflow_data_iterator,
)

matplotlib.use("Agg")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    now = get_current_time()
    assert args.use_cfg, "In this assignment, we train with CFG setup only."

    # Create save directory
    save_dir = Path(f"results/instaflow-{now}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load InstaFlow distillation dataset
    print(f"Loading distillation dataset from {args.distill_data_path}")
    distill_dataset = InstaFlowDataset(
        args.distill_data_path,
        use_cfg=args.use_cfg
    )

    train_dl = torch.utils.data.DataLoader(
        distill_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )
    train_it = get_instaflow_data_iterator(train_dl)

    # Get image resolution and num_classes from dataset metadata
    image_resolution = distill_dataset.metadata.get("image_resolution", 64)
    num_classes = distill_dataset.metadata.get("num_classes", None)

    print(f"Image resolution: {image_resolution}")
    if args.use_cfg:
        print(f"Number of classes: {num_classes}")

    # Initialize InstaFlow student network
    # Note: Can use a smaller "very simple U-Net" for faster inference
    network = UNet(
        image_resolution=image_resolution,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        num_classes=num_classes-1,  # Exclude null class
    )

    # Use InstaFlowModel instead of FlowMatching
    instaflow = InstaFlowModel(network, use_lpips=args.use_lpips)
    instaflow = instaflow.to(config.device)

    # Optimizer
    optimizer = torch.optim.Adam(instaflow.network.parameters(), lr=args.lr)

    print(f"Setting up LR scheduler: {config.warmup_steps} warmup steps, "
          f"then cosine decay for {config.train_num_steps - config.warmup_steps} steps.")

    # 1. Linear warmup scheduler
    # We'll start from a very small LR and warm up to the base LR
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6,  # Start LR = 1e-6 * args.lr
        end_factor=1.0,       # End LR = 1.0 * args.lr
        total_iters=config.warmup_steps
    )

    # 2. Cosine decay scheduler
    # This will run for the remaining steps after warmup
    # Ensure T_max is not negative if warmup_steps >= train_num_steps
    cosine_t_max = max(1, config.train_num_steps - config.warmup_steps)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_t_max,
        eta_min=0  # As in the original scheduler
    )

    # 3. Combine them sequentially
    # The scheduler will switch from warmup to cosine at `milestones`
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_steps]
    )

    step = 0
    losses = []
    print(f"Starting InstaFlow training for {config.train_num_steps} steps...")
    print("Goal: Learn to generate images in ONE STEP from noise to data")

    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            if step % config.log_interval == 0:
                instaflow.eval()
                # Plot loss curve
                plt.plot(losses)
                plt.xlabel('Training Step')
                plt.ylabel('Distillation Loss (L2 + LPIPS)' if args.use_lpips else 'Distillation Loss (L2)')
                plt.title('InstaFlow Training Loss')
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()

                # Generate sample images with ONE-STEP inference
                shape = (4, 3, instaflow.image_resolution, instaflow.image_resolution)
                if args.use_cfg:
                    class_label = torch.tensor([1, 1, 2, 3]).to(config.device)
                    # ONE-STEP GENERATION (no guidance scale needed - baked in!)
                    samples = instaflow.sample(
                        shape,
                        class_label=class_label
                    )
                else:
                    samples = instaflow.sample(shape)

                pil_images = tensor_to_pil_image(samples)
                for i, img in enumerate(pil_images):
                    img.save(save_dir / f"step={step}-{i}.png")

                # Save checkpoint
                instaflow.save(f"{save_dir}/last.ckpt")
                instaflow.train()

            # Load batch from distillation dataset
            if args.use_cfg:
                x_0, x_1, label = next(train_it)
                x_0, x_1, label = x_0.to(config.device), x_1.to(config.device), label.to(config.device)
            else:
                x_0, x_1 = next(train_it)
                x_0, x_1 = x_0.to(config.deveice), x_1.to(config.device)
                label = None

            # Compute InstaFlow distillation loss
            # The student learns to map x_0 directly to x_1 in ONE STEP
            # InstaFlowModel.get_loss handles the one-step objective (Eq. 6 in paper)
            if args.use_cfg:
                loss = instaflow.get_loss(x_1, x_0, class_label=label)
            else:
                loss = instaflow.get_loss(x_1, x_0)

            pbar.set_description(f"Loss: {loss.item():.4f} LR: {scheduler.get_last_lr()[0]:.6f}")

            # Optimization step
            optimizer.zero_grad()
            loss.backward()

            # Optional: Gradient clipping for stability
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(instaflow.network.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()  # This steps the SequentialLR
            losses.append(loss.item())

            step += 1
            pbar.update(1)

    print(f"Training completed! Final checkpoint saved at {save_dir}/last.ckpt")
    print("\n" + "="*80)
    print("Phase 2 complete! You now have an InstaFlow one-step generator.")
    print("\nGenerate images with ONE STEP:")
    print(f"python -m image_common.sampling \\")
    print(f"  --use_cfg \\")
    print(f"  --ckpt_path {save_dir}/last.ckpt \\")
    print(f"  --save_dir results/instaflow_samples \\")
    print(f"  --num_inference_steps 1")
    print("\nEvaluate your model:")
    print(f"python -m task4_instaflow.evaluate_instaflow \\")
    print(f"  --rf1_ckpt_path <2RF_CKPT> \\")
    print(f"  --instaflow_ckpt_path {save_dir}/last.ckpt \\")
    print(f"  --save_dir results/instaflow_eval")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Train InstaFlow one-step generator")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--distill_data_path", type=str, required=True,
                        help="Path to InstaFlow distillation dataset from Phase 2 data generation")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000,
        help="Number of training steps",
    )
    # Note: Added a default for warmup_steps, but it's often set via command line
    parser.add_argument("--warmup_steps", type=int, default=5000,
                        help="Number of linear warmup steps")
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.0)
    parser.add_argument("--use_lpips", action="store_true",
                        help="Use LPIPS perceptual loss instead of L2 (improves quality)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping (0 to disable)")
    args = parser.parse_args()
    main(args)