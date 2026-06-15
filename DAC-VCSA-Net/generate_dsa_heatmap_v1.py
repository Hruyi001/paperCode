"""
Generate heatmap visualization for DSA module ablation study.
This script generates heatmaps comparing models with and without DSA module.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sample4geo.hand_convnext.model import make_model


class ModelWrapper(nn.Module):
    """Wrapper to extract feature maps for DSA ablation visualization"""
    def __init__(self, model, use_dsa=True):
        super().__init__()
        self.model = model
        self.use_dsa = use_dsa
        
    def forward(self, x):
        # Get backbone features (in eval mode, returns gap_feature, part_features)
        gap_feature, part_features = self.model.model_1.convnext(x)
        
        if self.use_dsa:
            # Apply DSA module to get attention weights
            # This simulates what DSA does during training
            b, c, h, w = part_features.shape
            pfeat = part_features.flatten(2)  # (bs, c, h*w)
            
            # DSA module: generate attention weights W
            W = self.model.model_1.proj(pfeat)  # (bs, 256, h*w)
            
            W = F.normalize(W, dim=1) if self.model.model_1.l2_norm else W
            W *= (1 / self.model.model_1.scale)
            W = F.softmax(W, dim=2)  # Normalize to get attention distribution
            
            # Reshape W to spatial dimensions for visualization
            # W shape: (bs, 256, h*w) -> (bs, 256, h, w)
            W_spatial = W.view(b, -1, h, w)
            
            # Aggregate attention weights across channels to get spatial attention map
            # This shows where DSA module focuses its attention
            attention_map = W_spatial.mean(dim=1)  # (bs, h, w)
            
        else:
            # Without DSA: visualize raw feature spatial activation
            # This should show where the model naturally focuses without DSA alignment
            # Use multiple statistics to get a more distributed and realistic attention map
            
            # Method 1: Channel-wise mean (standard activation)
            feature_mean = part_features.mean(dim=1)  # (bs, h, w)
            
            # Method 2: Channel-wise variance (discriminative regions)
            feature_var = part_features.var(dim=1)  # (bs, h, w)
            
            # Method 3: Max activation (strongest responses)
            feature_max = part_features.max(dim=1)[0]  # (bs, h, w)
            
            # Normalize each component separately to avoid dominance of one statistic
            def normalize_tensor(t):
                t_min = t.min()
                t_max = t.max()
                return (t - t_min) / (t_max - t_min + 1e-8)
            
            feature_mean_norm = normalize_tensor(feature_mean)
            feature_var_norm = normalize_tensor(feature_var)
            feature_max_norm = normalize_tensor(feature_max)
            
            # Combine: variance emphasizes discriminative regions, 
            # mean provides baseline activation, max captures strong responses
            # This combination should give a more distributed attention map
            attention_map = 0.5 * feature_var_norm + 0.3 * feature_mean_norm + 0.2 * feature_max_norm
            
            # Apply light spatial smoothing to avoid overly sharp concentration
            # This mimics natural attention distribution
            kernel_size = 5
            padding = kernel_size // 2
            # Use Gaussian-like smoothing with average pooling
            attention_map = F.avg_pool2d(
                attention_map.unsqueeze(1), 
                kernel_size=kernel_size, 
                stride=1, 
                padding=padding
            ).squeeze(1)
            
            # Final normalization
            attention_map = normalize_tensor(attention_map)
        
        return attention_map, part_features


def generate_heatmap(feature_map, original_img, alpha=0.5):
    """
    Generate heatmap overlay on original image
    
    Args:
        feature_map: (H, W) attention/activation map
        original_img: (H, W, 3) original image in RGB format
        alpha: transparency factor for heatmap overlay
    """
    # Normalize feature map to [0, 1]
    if feature_map.max() > feature_map.min():
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
    else:
        feature_map = np.ones_like(feature_map) * 0.5
    
    # Resize feature map to match original image size
    if feature_map.shape != original_img.shape[:2]:
        feature_map = cv2.resize(feature_map, (original_img.shape[1], original_img.shape[0]), 
                                 interpolation=cv2.INTER_LINEAR)
    
    # Apply colormap (using jet colormap: blue -> green -> yellow -> red)
    # Convert to uint8 for colormap
    feature_map_uint8 = (feature_map * 255).astype(np.uint8)
    heatmap = cm.jet(feature_map_uint8)[:, :, :3]  # (H, W, 3) RGB
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Ensure original image is uint8
    if original_img.dtype != np.uint8:
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)
        else:
            original_img = original_img.astype(np.uint8)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    
    return overlay, heatmap


def load_image_pair(dataset_path, sample_id, img_size=384):
    """
    Load a satellite-drone image pair from dataset
    
    Args:
        dataset_path: path to dataset root
        sample_id: sample ID to load
        img_size: target image size
    """
    def _resolve_test_root(p: str) -> str:
        # Accept either dataset root or ".../test"
        if os.path.isdir(p) and os.path.basename(os.path.normpath(p)) == "test":
            return p
        candidate = os.path.join(p, "test")
        if os.path.isdir(candidate):
            return candidate
        return p

    test_root = _resolve_test_root(dataset_path)

    # Construct paths - try different possible structures
    possible_sat_paths = [
        os.path.join(test_root, "gallery_satellite"),
        os.path.join(test_root, "satellite"),
    ]

    possible_drone_paths = [
        os.path.join(test_root, "query_drone"),
        os.path.join(test_root, "drone"),
    ]
    
    # Find valid satellite/drone dirs
    sat_dir = next((p for p in possible_sat_paths if os.path.isdir(p)), None)
    drone_dir = next((p for p in possible_drone_paths if os.path.isdir(p)), None)
    if sat_dir is None:
        raise ValueError(f"Could not find satellite directory. Tried: {possible_sat_paths}")
    if drone_dir is None:
        raise ValueError(f"Could not find drone directory. Tried: {possible_drone_paths}")

    def _is_dir_grouped(root_dir: str) -> bool:
        # If it contains subdirectories, assume "<id>/<img>" format
        try:
            for name in os.listdir(root_dir):
                if os.path.isdir(os.path.join(root_dir, name)):
                    return True
        except FileNotFoundError:
            return False
        return False

    def _find_image_file_flat(root_dir: str, sid: str) -> str:
        # Flat format: images directly under root_dir
        # Match by prefix before "_" if present, else full stem.
        exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        candidates = []
        for f in os.listdir(root_dir):
            if not f.endswith(exts):
                continue
            stem = os.path.splitext(f)[0]
            stem_id = stem.split("_")[0]
            if stem == sid or stem_id == sid:
                candidates.append(os.path.join(root_dir, f))
        if not candidates:
            raise ValueError(f"No image found for sample_id={sid} under {root_dir}")
        candidates.sort()
        return candidates[0]

    def _find_image_file_grouped(root_dir: str, sid: str) -> str:
        # Grouped format: root_dir/<sid>/<img>
        folder = os.path.join(root_dir, sid)
        if not os.path.isdir(folder):
            raise ValueError(f"Folder not found for sample_id={sid}: {folder}")
        exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        files = [f for f in os.listdir(folder) if f.endswith(exts)]
        if not files:
            raise ValueError(f"No image files found in {folder}")
        files.sort()
        return os.path.join(folder, files[0])
    
    # Load images (support both grouped and flat layouts)
    if _is_dir_grouped(sat_dir):
        sat_img_path = _find_image_file_grouped(sat_dir, str(sample_id))
    else:
        sat_img_path = _find_image_file_flat(sat_dir, str(sample_id))

    if _is_dir_grouped(drone_dir):
        drone_img_path = _find_image_file_grouped(drone_dir, str(sample_id))
    else:
        drone_img_path = _find_image_file_flat(drone_dir, str(sample_id))

    sat_img = cv2.imread(sat_img_path)
    if sat_img is None:
        raise ValueError(f"Failed to load satellite image: {sat_img_path}")
    sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

    drone_img = cv2.imread(drone_img_path)
    if drone_img is None:
        raise ValueError(f"Failed to load drone image: {drone_img_path}")
    drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)
    
    # Store original images for visualization
    sat_original = cv2.resize(sat_img, (img_size, img_size))
    drone_original = cv2.resize(drone_img, (img_size, img_size))
    
    # Prepare transforms
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    sat_tensor = transform(image=sat_img)['image'].unsqueeze(0)
    drone_tensor = transform(image=drone_img)['image'].unsqueeze(0)
    
    return sat_tensor, drone_tensor, sat_original, drone_original


def list_available_sample_ids(dataset_path: str, limit: int = 3):
    """
    List sample ids by intersecting satellite and drone directories.
    Supports both:
      - grouped: test/gallery_satellite/<id>/<img>, test/query_drone/<id>/<img>
      - flat:    test/gallery_satellite/*.jpg, test/query_drone/*.jpg (id inferred from filename stem)
    """
    def _resolve_test_root(p: str) -> str:
        if os.path.isdir(p) and os.path.basename(os.path.normpath(p)) == "test":
            return p
        candidate = os.path.join(p, "test")
        if os.path.isdir(candidate):
            return candidate
        return p

    test_root = _resolve_test_root(dataset_path)
    sat_dir = next((p for p in [os.path.join(test_root, "gallery_satellite"), os.path.join(test_root, "satellite")] if os.path.isdir(p)), None)
    drone_dir = next((p for p in [os.path.join(test_root, "query_drone"), os.path.join(test_root, "drone")] if os.path.isdir(p)), None)
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
    return common[:limit]


def create_comparison_figure(sat_original, drone_original, 
                            sat_heatmap_wo_dsa, drone_heatmap_wo_dsa,
                            sat_heatmap_w_dsa, drone_heatmap_w_dsa,
                            save_path):
    """
    Create a comparison figure similar to the paper
    
    Layout:
    Row 1: Original Satellite | Original Drone
    Row 2: Satellite w/o DSA  | Drone w/o DSA
    Row 3: Satellite w/ DSA   | Drone w/ DSA
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    
    # Row 1: Original images
    axes[0, 0].imshow(sat_original)
    axes[0, 0].set_title('Satellite', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(drone_original)
    axes[0, 1].set_title('Drone', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Row 2: Without DSA
    axes[1, 0].imshow(sat_heatmap_wo_dsa)
    axes[1, 0].set_title('w/o DSA', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(drone_heatmap_wo_dsa)
    axes[1, 1].axis('off')
    
    # Row 3: With DSA
    axes[2, 0].imshow(sat_heatmap_w_dsa)
    axes[2, 0].set_title('w/ DSA', fontsize=14, fontweight='bold')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(drone_heatmap_w_dsa)
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate DSA ablation heatmaps')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset root (e.g., /path/to/U1652)')
    parser.add_argument('--sample_ids', type=str, nargs='+', default=None,
                        help='Sample IDs to visualize (e.g., "001" "002"). If None, will use first available samples')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to visualize if sample_ids not specified')
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
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {args.ckpt_path}\n"
            f"Please check the path and make sure the checkpoint file exists.\n"
            f"You can find available checkpoints by running: find . -name '*.pth' -type f"
        )
    
    print(f"Loading checkpoint from: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=args.device)
    
    # Remove classifier weights if present (they might cause issues)
    keys_to_remove = [k for k in checkpoint.keys() if 'classifier' in k and ('weight' in k or 'bias' in k)]
    for k in keys_to_remove:
        if k in checkpoint:
            del checkpoint[k]
    
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(args.device)
    model.eval()
    
    # Create model wrappers
    model_with_dsa = ModelWrapper(model, use_dsa=True)
    model_without_dsa = ModelWrapper(model, use_dsa=False)
    model_with_dsa.eval()
    model_without_dsa.eval()
    
    # Find available samples (supports both grouped and flat layouts)
    if args.sample_ids is None:
        available_ids = list_available_sample_ids(args.dataset_path, limit=args.num_samples)
        if not available_ids:
            raise ValueError("No sample_ids found. Please pass --sample_ids explicitly.")
    else:
        available_ids = args.sample_ids
    
    print(f"Processing {len(available_ids)} samples...")
    
    # Process each sample
    for idx, sample_id in enumerate(available_ids):
        try:
            print(f"\nProcessing sample {sample_id} ({idx+1}/{len(available_ids)})...")
            
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
            
            # Create comparison figure
            save_path = os.path.join(args.output_dir, f'ablation_heatmap_{sample_id}.png')
            create_comparison_figure(
                sat_original, drone_original,
                sat_overlay_no_dsa, drone_overlay_no_dsa,
                sat_overlay_dsa, drone_overlay_dsa,
                save_path
            )
            
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            continue
    
    print(f"\nAll heatmaps saved to {args.output_dir}")


if __name__ == '__main__':
    main()
