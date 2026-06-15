#!/usr/bin/env python3
"""
Generate heatmaps for all test images in the dataset.
This script processes all images in the test dataset and generates heatmaps.
"""

import os
import sys
import argparse
from tqdm import tqdm
from generate_dsa_heatmap import (
    load_image_pair, 
    generate_heatmap, 
    create_comparison_figure,
    ModelWrapper
)
import torch
import torch.nn as nn
from sample4geo.hand_convnext.model import make_model


def list_all_sample_ids(dataset_path: str):
    """
    List all available sample IDs in the test dataset.
    """
    def _resolve_test_root(p: str) -> str:
        if os.path.isdir(p) and os.path.basename(os.path.normpath(p)) == "test":
            return p
        candidate = os.path.join(p, "test")
        if os.path.isdir(candidate):
            return candidate
        return p

    test_root = _resolve_test_root(dataset_path)
    sat_dir = next((p for p in [
        os.path.join(test_root, "gallery_satellite"),
        os.path.join(test_root, "satellite")
    ] if os.path.isdir(p)), None)
    
    drone_dir = next((p for p in [
        os.path.join(test_root, "query_drone"),
        os.path.join(test_root, "drone")
    ] if os.path.isdir(p)), None)
    
    if sat_dir is None or drone_dir is None:
        raise ValueError(f"Could not find sat/drone dirs under {test_root}")

    def _is_dir_grouped(root_dir: str) -> bool:
        return any(os.path.isdir(os.path.join(root_dir, n)) for n in os.listdir(root_dir))

    def _ids_from_grouped(root_dir: str):
        return {n for n in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, n))}

    def _ids_from_flat(root_dir: str):
        exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        ids = set()
        for f in os.listdir(root_dir):
            if not f.endswith(exts):
                continue
            stem = os.path.splitext(f)[0]
            ids.add(stem.split("_")[0])
        return ids

    sat_ids = _ids_from_grouped(sat_dir) if _is_dir_grouped(sat_dir) else _ids_from_flat(sat_dir)
    drone_ids = _ids_from_grouped(drone_dir) if _is_dir_grouped(drone_dir) else _ids_from_flat(drone_dir)

    common = sorted(list(sat_ids.intersection(drone_ids)))
    return common


def main():
    parser = argparse.ArgumentParser(description='Generate heatmaps for all test images')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset root (e.g., /root/dataset/University-Release)')
    parser.add_argument('--img_size', type=int, default=384,
                        help='Image size')
    parser.add_argument('--output_dir', type=str, default='./heatmap_img',
                        help='Output directory for heatmaps')
    parser.add_argument('--nclasses', type=int, default=701,
                        help='Number of classes')
    parser.add_argument('--block', type=int, default=2,
                        help='Number of blocks')
    parser.add_argument('--triplet_loss', type=float, default=0.3,
                        help='Triplet loss weight')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing (default: 1 for individual heatmaps)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model configuration
    class Config:
        def __init__(self):
            self.nclasses = args.nclasses
            self.block = args.block
            self.triplet_loss = args.triplet_loss
            self.resnet = False
            self.views = 2
    
    config = Config()
    
    # Load model
    print("Loading model...")
    model = make_model(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt_path}...")
    checkpoint = torch.load(args.ckpt_path, map_location=args.device)
    
    # Handle DataParallel wrapper
    if 'module.' in list(checkpoint.keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        checkpoint = new_state_dict
    
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(args.device)
    model.eval()
    
    # Create model wrappers
    model_with_dsa = ModelWrapper(model, use_dsa=True)
    model_without_dsa = ModelWrapper(model, use_dsa=False)
    model_with_dsa.eval()
    model_without_dsa.eval()
    
    # Get all sample IDs
    print(f"Scanning dataset at {args.dataset_path}...")
    try:
        all_sample_ids = list_all_sample_ids(args.dataset_path)
        print(f"Found {len(all_sample_ids)} sample pairs to process")
    except Exception as e:
        print(f"Error scanning dataset: {e}")
        sys.exit(1)
    
    if len(all_sample_ids) == 0:
        print("No sample pairs found in the dataset!")
        sys.exit(1)
    
    # Process each sample with progress bar
    successful = 0
    failed = 0
    failed_samples = []
    
    print(f"\nStarting batch processing of {len(all_sample_ids)} samples...")
    print(f"Output directory: {args.output_dir}\n")
    
    # Use tqdm for progress bar
    for sample_id in tqdm(all_sample_ids, desc="Generating heatmaps", unit="image"):
        try:
            # Load image pair
            sat_tensor, drone_tensor, sat_original, drone_original = load_image_pair(
                args.dataset_path, sample_id, args.img_size
            )
            sat_tensor = sat_tensor.to(args.device)
            drone_tensor = drone_tensor.to(args.device)
            
            # Generate heatmaps with DSA
            with torch.no_grad():
                sat_attn_dsa, _ = model_with_dsa(sat_tensor)
                drone_attn_dsa, _ = model_with_dsa(drone_tensor)
                
                sat_attn_no_dsa, _ = model_without_dsa(sat_tensor)
                drone_attn_no_dsa, _ = model_without_dsa(drone_tensor)
            
            # Convert to numpy
            sat_attn_dsa = sat_attn_dsa[0].cpu().numpy()
            drone_attn_dsa = drone_attn_dsa[0].cpu().numpy()
            sat_attn_no_dsa = sat_attn_no_dsa[0].cpu().numpy()
            drone_attn_no_dsa = drone_attn_no_dsa[0].cpu().numpy()
            
            # Generate heatmap overlays
            sat_overlay_no_dsa, _ = generate_heatmap(sat_attn_no_dsa, sat_original)
            drone_overlay_no_dsa, _ = generate_heatmap(drone_attn_no_dsa, drone_original)
            sat_overlay_dsa, _ = generate_heatmap(sat_attn_dsa, sat_original)
            drone_overlay_dsa, _ = generate_heatmap(drone_attn_dsa, drone_original)
            
            # Create comparison figure (suppress print output for cleaner progress bar)
            save_path = os.path.join(args.output_dir, f'heatmap_{sample_id}.png')
            # Temporarily redirect stdout to suppress print from create_comparison_figure
            import contextlib
            import io
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                create_comparison_figure(
                    sat_original, drone_original,
                    sat_overlay_no_dsa, drone_overlay_no_dsa,
                    sat_overlay_dsa, drone_overlay_dsa,
                    save_path
                )
            
            successful += 1
            
        except Exception as e:
            failed += 1
            failed_samples.append((sample_id, str(e)))
            tqdm.write(f"  ✗ Error processing {sample_id}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Batch Processing Completed!")
    print(f"{'='*60}")
    print(f"  Total samples: {len(all_sample_ids)}")
    print(f"  ✓ Successful: {successful} ({successful/len(all_sample_ids)*100:.1f}%)")
    print(f"  ✗ Failed: {failed} ({failed/len(all_sample_ids)*100:.1f}%)")
    print(f"  Output directory: {args.output_dir}")
    
    if failed_samples:
        print(f"\nFailed samples:")
        for sample_id, error in failed_samples[:10]:  # Show first 10 failures
            print(f"  - {sample_id}: {error[:80]}")
        if len(failed_samples) > 10:
            print(f"  ... and {len(failed_samples) - 10} more")
    
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
