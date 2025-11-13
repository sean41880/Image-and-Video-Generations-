"""
Generate InstaFlow Distillation Dataset (Task 4 - Phase 2)

This script generates training pairs (x_0, x_1) for distilling a 2-Rectified Flow teacher model
into a one-step generator. The 1-RF teacher (from Phase 1) has straighter generation paths,
making it an ideal teacher for one-step distillation.

Uses lower CFG guidance scale (α₂ = 1.5) compared to Phase 1 (α₁ = 7.5) to avoid over-saturation.

Reference: Liu et al., "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append('.')
from image_common.dataset import tensor_to_pil_image
from image_common.fm import FlowMatching

def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save configuration
    config = vars(args)
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    device = f"cuda:{args.gpu}"

    # Load 2-Rectified Flow teacher model (from Phase 1)
    print(f"Loading 2-Rectified Flow teacher checkpoint from {args.rf1_ckpt_path}")
    teacher = FlowMatching(None, None)
    teacher.load(args.rf1_ckpt_path)
    teacher.eval()
    teacher = teacher.to(device)

    if args.use_cfg:
        assert teacher.network.use_cfg, "The 1-RF model was not trained with CFG support."
        num_classes = teacher.network.class_embedding.num_embeddings
        print(f"Using CFG with {num_classes} classes (including null class)")
    else:
        num_classes = None

    total_num_samples = args.num_samples
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    print(f"\nGenerating {total_num_samples} distillation pairs for InstaFlow...")
    print(f"1-RF teacher uses {args.num_inference_steps} ODE steps")
    print(f"CFG guidance scale (α₂): {args.cfg_scale}")
    print(f"This is Phase 2: Distilling to one-step generator\n")

    for i in tqdm(range(num_batches), desc="Generating InstaFlow dataset"):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx

        # Sample initial noise
        shape = (B, 3, teacher.image_resolution, teacher.image_resolution)
        x_0 = torch.randn(shape).to(device)

        # Sample class labels if using CFG
        if args.use_cfg:
            labels = torch.randint(1, num_classes, (B,)).to(device)
        else:
            labels = None

        # Generate x_1 using 1-RF teacher model with CFG (guidance scale α₂)
        x_1 = teacher.sample(
            shape,
            class_label=labels,
            guidance_scale=args.cfg_scale,
            num_inference_timesteps=args.num_inference_steps,
            verbose=False
        )

        # Save the pairs to disk
        for j in range(B):
            sample_idx = sidx + j

            torch.save(x_0[j].cpu(), save_dir / f"{sample_idx:06d}_x0.pt")
            torch.save(x_1[j].cpu(), save_dir / f"{sample_idx:06d}_x1.pt")

            if args.use_cfg:
                torch.save(labels[j].cpu(), save_dir / f"{sample_idx:06d}_label.pt")

            if args.save_images and sample_idx < 100:
                img_x1 = tensor_to_pil_image(x_1[j].cpu(), single_image=True)
                img_x1.save(save_dir / f"{sample_idx:06d}_x1.png")

    print(f"InstaFlow distillation dataset saved to {save_dir}")
    print(f"Total samples: {total_num_samples}")

    metadata = {
        "num_samples": total_num_samples,
        "teacher_type": "2-rectified-flow",
        "teacher_inference_steps": args.num_inference_steps,
        "use_cfg": args.use_cfg,
        "teacher_cfg_scale": args.cfg_scale if args.use_cfg else None,
        "image_resolution": teacher.image_resolution,
        "num_classes": num_classes if args.use_cfg else None,
        "distillation_type": "instaflow",
        "phase": "phase2_instaflow_training",
        "target_steps": 1,
        "purpose": "Distill 1-RF teacher to one-step InstaFlow student",
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("Dataset generation complete!")
    print(f"Next step: Train InstaFlow one-step generator with:")
    print(f"python -m task4_instaflow.train_instaflow")
    print(f"  --distill_data_path {save_dir}")
    if args.use_cfg:
        print(f"  --use_cfg")
    print(f"  --train_num_steps 100000")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Generate InstaFlow distillation dataset from 1-RF teacher"
    )
    parser.add_argument("--rf1_ckpt_path", type=str, required=True,
                        help="Path to 2-Rectified Flow teacher checkpoint from Phase 1 (.ckpt file)")
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of distillation pairs to generate (default: 50000)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for generation")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--save_dir", type=str, default="data/afhq_instaflow",
                        help="Directory to save InstaFlow distillation dataset")
    parser.add_argument("--use_cfg", action="store_true",
                        help="Use classifier-free guidance")
    parser.add_argument("--cfg_scale", type=float, default=1.5,
                        help="CFG guidance scale α₂ for 1-RF teacher (default: 1.5, lower than Phase 1)")
    parser.add_argument("--num_inference_steps", type=int, default=20,
                        help="Number of ODE steps for 1-RF teacher sampling (default: 20)")
    parser.add_argument("--save_images", action="store_true",
                        help="Save first 100 samples as images for inspection")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)