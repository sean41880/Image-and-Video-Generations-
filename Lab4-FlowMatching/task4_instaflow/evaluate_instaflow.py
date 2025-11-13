"""
Evaluation Script for InstaFlow One-Step Generator (Task 4 - Phase 2)

Compares InstaFlow one-step generation against 2-Rectified Flow teacher model.
Measures both quality (FID) and speed (samples/second).

Usage:
    python -m task4_instaflow.evaluate_instaflow \
        --rf1_ckpt_path results/2rf_from_ddpm-XXX/last.ckpt \
        --instaflow_ckpt_path results/instaflow-XXX/last.ckpt \
        --save_dir results/instaflow_eval
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('.')
from image_common.dataset import tensor_to_pil_image
from image_common.fm import FlowMatching
from image_common.instaflow import InstaFlowModel


def generate_samples_fm(fm, num_samples, num_steps, batch_size, device, use_cfg=True, cfg_scale=7.5, model_name="Model"):
    """Generate samples from Flow Matching model with specified number of steps"""
    num_batches = int(np.ceil(num_samples / batch_size))
    all_samples = []

    start_time = time.time()

    for i in tqdm(range(num_batches), desc=f"{model_name} ({num_steps} step{'s' if num_steps > 1 else ''})"):
        sidx = i * batch_size
        eidx = min(sidx + batch_size, num_samples)
        B = eidx - sidx

        shape = (B, 3, fm.image_resolution, fm.image_resolution)

        if use_cfg:
            class_label = torch.randint(1, 4, (B,)).to(device)
            samples = fm.sample(
                shape,
                num_inference_timesteps=num_steps,
                class_label=class_label,
                guidance_scale=cfg_scale,
            )
        else:
            samples = fm.sample(shape, num_inference_timesteps=num_steps)

        all_samples.append(samples.cpu())

    elapsed_time = time.time() - start_time
    all_samples = torch.cat(all_samples, dim=0)[:num_samples]

    return all_samples, elapsed_time


def generate_samples_instaflow(instaflow, num_samples, batch_size, device, use_cfg=True, model_name="InstaFlow"):
    """Generate samples from InstaFlow one-step model"""
    num_batches = int(np.ceil(num_samples / batch_size))
    all_samples = []

    start_time = time.time()

    for i in tqdm(range(num_batches), desc=f"{model_name} (1 step)"):
        sidx = i * batch_size
        eidx = min(sidx + batch_size, num_samples)
        B = eidx - sidx

        shape = (B, 3, instaflow.image_resolution, instaflow.image_resolution)

        if use_cfg:
            class_label = torch.randint(1, 4, (B,)).to(device)
            # ONE-STEP GENERATION (CFG baked in!)
            samples = instaflow.sample(
                shape,
                class_label=class_label
            )
        else:
            samples = instaflow.sample(shape)

        all_samples.append(samples.cpu())

    elapsed_time = time.time() - start_time
    all_samples = torch.cat(all_samples, dim=0)[:num_samples]

    return all_samples, elapsed_time


def save_samples(samples, save_dir, prefix="sample"):
    """Save generated samples as images"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    pil_images = tensor_to_pil_image(samples)
    for i, img in enumerate(pil_images):
        img.save(save_dir / f"{prefix}_{i:04d}.png")


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"

    # Load 2-Rectified Flow teacher model
    print(f"Loading 2-Rectified Flow teacher model from {args.rf1_ckpt_path}")
    rf1_teacher = FlowMatching(None, None)
    rf1_teacher.load(args.rf1_ckpt_path)
    rf1_teacher.eval()
    rf1_teacher = rf1_teacher.to(device)

    print(f"Loading InstaFlow model from {args.instaflow_ckpt_path}")
    instaflow = InstaFlowModel(None)
    instaflow.load(args.instaflow_ckpt_path)
    instaflow.eval()
    instaflow = instaflow.to(device)

    # Evaluation results
    results = {}

    print(f"\n{'='*80}")
    print(f"InstaFlow One-Step Generation Evaluation")
    print(f"Generating {args.num_samples} samples")
    print(f"{'='*80}\n")

    # 1. 2-Rectified Flow Teacher (multi-step baseline)
    rf1_steps = args.rf1_inference_steps
    print(f"1. 2-Rectified Flow Teacher ({rf1_steps} steps)")
    samples_rf1, time_rf1 = generate_samples_fm(
        rf1_teacher, args.num_samples, rf1_steps, args.batch_size, device,
        use_cfg=args.use_cfg, cfg_scale=1.5, model_name="1-RF Teacher"
    )
    rf1_save_dir = save_dir / f"2rf_{rf1_steps}steps"
    save_samples(samples_rf1, rf1_save_dir)

    results[f'1-RF Teacher ({rf1_steps} steps)'] = {
        'steps': rf1_steps,
        'time': time_rf1,
        'samples_per_sec': args.num_samples / time_rf1,
        'speedup': 1.0,
        'save_dir': str(rf1_save_dir)
    }
    print(f"   Time: {time_rf1:.2f}s ({args.num_samples/time_rf1:.2f} samples/s)")
    print(f"   Saved to: {rf1_save_dir}\n")

    # 2. InstaFlow with 1 step (ONE-STEP GENERATION!)
    print("2. InstaFlow (ONE STEP)")
    samples_instaflow, time_instaflow = generate_samples_instaflow(
        instaflow, args.num_samples, args.batch_size, device,
        use_cfg=args.use_cfg, model_name="InstaFlow"
    )
    instaflow_save_dir = save_dir / "instaflow_1step"
    save_samples(samples_instaflow, instaflow_save_dir)

    speedup_instaflow = time_rf1 / time_instaflow
    results['InstaFlow (1 step)'] = {
        'steps': 1,
        'time': time_instaflow,
        'samples_per_sec': args.num_samples / time_instaflow,
        'speedup': speedup_instaflow,
        'save_dir': str(instaflow_save_dir)
    }
    print(f"   Time: {time_instaflow:.2f}s ({args.num_samples/time_instaflow:.2f} samples/s)")
    print(f"   Speedup: {speedup_instaflow:.2f}x 🚀")
    print(f"   Saved to: {instaflow_save_dir}\n")

    # Save results
    results_file = save_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Create comparison plots
    plot_comparison(results, save_dir)

    # Print FID measurement commands
    print("\n" + "="*80)
    print("To measure FID scores, run the following commands:")
    print("="*80)
    for model_name, model_results in results.items():
        print(f"\n# {model_name}:")
        print(f"python fid/measure_fid.py data/afhq/eval {model_results['save_dir']}")

    print("\n" + "="*80)
    print(f"Results saved to {results_file}")
    print("="*80)


def plot_comparison(results, save_dir):
    """Create comparison plots"""

    # Extract data
    models = list(results.keys())
    steps = [results[m]['steps'] for m in models]
    times = [results[m]['time'] for m in models]
    speedups = [results[m]['speedup'] for m in models]
    samples_per_sec = [results[m]['samples_per_sec'] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Number of steps
    colors = ['#3498db', '#2ecc71'][:len(models)]
    axes[0].bar(range(len(models)), steps, color=colors)
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=15, ha='right')
    axes[0].set_ylabel('Number of Inference Steps')
    axes[0].set_title('Inference Steps Comparison')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, v in enumerate(steps):
        axes[0].text(i, v + max(steps)*0.02, str(v), ha='center', va='bottom', fontweight='bold')

    # Plot 2: Sampling time
    axes[1].bar(range(len(models)), times, color=colors)
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, rotation=15, ha='right')
    axes[1].set_ylabel('Sampling Time (seconds)')
    axes[1].set_title('Generation Speed Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(times):
        axes[1].text(i, v + max(times)*0.02, f'{v:.1f}s', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Speedup
    axes[2].bar(range(len(models)), speedups, color=colors)
    axes[2].set_xticks(range(len(models)))
    axes[2].set_xticklabels(models, rotation=15, ha='right')
    axes[2].set_ylabel('Speedup vs 1-RF Teacher')
    axes[2].set_title('Speedup Comparison')
    axes[2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='1-RF Baseline')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].legend()

    # Add value labels
    for i, v in enumerate(speedups):
        axes[2].text(i, v + max(speedups)*0.02, f'{v:.1f}x', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plot_path = save_dir / "instaflow_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Evaluate InstaFlow one-step generation against 1-RF teacher")
    parser.add_argument("--rf1_ckpt_path", type=str, required=True,
                        help="Path to 2-Rectified Flow teacher checkpoint from Phase 1 (.ckpt file)")
    parser.add_argument("--instaflow_ckpt_path", type=str, required=True,
                        help="Path to InstaFlow checkpoint from Phase 2")
    parser.add_argument("--save_dir", type=str, default="results/instaflow_evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples to generate for evaluation")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for generation")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--use_cfg", action="store_true", default=True,
                        help="Use classifier-free guidance")
    parser.add_argument("--rf1_inference_steps", type=int, default=20,
                        help="Number of ODE steps for 1-RF teacher sampling")

    args = parser.parse_args()
    main(args)
