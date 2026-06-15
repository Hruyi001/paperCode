"""
Generate VCSA-Net heatmaps comparing backbone features with VRM/SAM/CSM-enhanced features.
"""

import argparse
import os
import sys
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2

DEFAULT_REPO_ROOT = str(Path(__file__).resolve().parents[1])
DEFAULT_CKPT_PATH = os.path.join(
    DEFAULT_REPO_ROOT,
    "checkpoints/university-close-d2s-drop5/convnext_base.fb_in22k_ft_in1k_384/0613202924/weights_end.pth",
)

repo_root = os.environ.get("VCSA_NET_REPO_ROOT", DEFAULT_REPO_ROOT)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sample4geo.hand_convnext.model import make_model


class ModelWrapper(nn.Module):
    def __init__(self, model, use_vcsa=True):
        super().__init__()
        self.model = model
        self.use_vcsa = use_vcsa

    def forward(self, x):
        backbone = self.model.model_1
        _, raw_part_features = backbone.convnext(x)
        part_features = raw_part_features
        if self.use_vcsa:
            part_features = backbone.enhanced_part_features(raw_part_features).contiguous()

        feature_mean = part_features.mean(dim=1)
        feature_var = part_features.var(dim=1)
        feature_max = part_features.max(dim=1)[0]

        def normalize_tensor(t):
            t_min = t.amin(dim=(-2, -1), keepdim=True)
            t_max = t.amax(dim=(-2, -1), keepdim=True)
            return (t - t_min) / (t_max - t_min + 1e-8)

        attention_map = (
            0.5 * normalize_tensor(feature_var)
            + 0.3 * normalize_tensor(feature_mean)
            + 0.2 * normalize_tensor(feature_max)
        )
        attention_map = F.avg_pool2d(attention_map.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        attention_map = normalize_tensor(attention_map)
        return attention_map, part_features


def generate_heatmap(feature_map, original_img, alpha=0.5):
    if feature_map.max() > feature_map.min():
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
    else:
        feature_map = np.ones_like(feature_map) * 0.5

    if feature_map.shape != original_img.shape[:2]:
        feature_map = cv2.resize(feature_map, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)

    feature_map_uint8 = (feature_map * 255).astype(np.uint8)
    heatmap = cm.jet(feature_map_uint8)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    if original_img.dtype != np.uint8:
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)
        else:
            original_img = original_img.astype(np.uint8)

    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    return overlay, heatmap


def load_image_pair(dataset_path, sample_id, img_size=384):
    def resolve_test_root(path):
        if os.path.isdir(path) and os.path.basename(os.path.normpath(path)) == "test":
            return path
        candidate = os.path.join(path, "test")
        if os.path.isdir(candidate):
            return candidate
        return path

    test_root = resolve_test_root(dataset_path)
    possible_sat_paths = [
        os.path.join(test_root, "gallery_satellite"),
        os.path.join(test_root, "satellite"),
    ]
    possible_drone_paths = [
        os.path.join(test_root, "query_drone"),
        os.path.join(test_root, "drone"),
    ]

    sat_dir = next((p for p in possible_sat_paths if os.path.isdir(p)), None)
    drone_dir = next((p for p in possible_drone_paths if os.path.isdir(p)), None)
    if sat_dir is None:
        raise ValueError(f"Could not find satellite directory. Tried: {possible_sat_paths}")
    if drone_dir is None:
        raise ValueError(f"Could not find drone directory. Tried: {possible_drone_paths}")

    def is_dir_grouped(root_dir):
        return any(os.path.isdir(os.path.join(root_dir, name)) for name in os.listdir(root_dir))

    def find_image_file_flat(root_dir, sid):
        exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        candidates = []
        for filename in os.listdir(root_dir):
            if not filename.endswith(exts):
                continue
            stem = os.path.splitext(filename)[0]
            stem_id = stem.split("_")[0]
            if stem == sid or stem_id == sid:
                candidates.append(os.path.join(root_dir, filename))
        if not candidates:
            raise ValueError(f"No image found for sample_id={sid} under {root_dir}")
        candidates.sort()
        return candidates[0]

    def find_image_file_grouped(root_dir, sid):
        folder = os.path.join(root_dir, sid)
        if not os.path.isdir(folder):
            raise ValueError(f"Folder not found for sample_id={sid}: {folder}")
        exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        files = [filename for filename in os.listdir(folder) if filename.endswith(exts)]
        if not files:
            raise ValueError(f"No image files found in {folder}")
        files.sort()
        return os.path.join(folder, files[0])

    sat_img_path = find_image_file_grouped(sat_dir, str(sample_id)) if is_dir_grouped(sat_dir) else find_image_file_flat(sat_dir, str(sample_id))
    drone_img_path = find_image_file_grouped(drone_dir, str(sample_id)) if is_dir_grouped(drone_dir) else find_image_file_flat(drone_dir, str(sample_id))

    sat_img = cv2.imread(sat_img_path)
    if sat_img is None:
        raise ValueError(f"Failed to load satellite image: {sat_img_path}")
    sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

    drone_img = cv2.imread(drone_img_path)
    if drone_img is None:
        raise ValueError(f"Failed to load drone image: {drone_img_path}")
    drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)

    sat_original = cv2.resize(sat_img, (img_size, img_size))
    drone_original = cv2.resize(drone_img, (img_size, img_size))

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    sat_tensor = transform(image=sat_img)["image"].unsqueeze(0)
    drone_tensor = transform(image=drone_img)["image"].unsqueeze(0)
    return sat_tensor, drone_tensor, sat_original, drone_original


def list_available_sample_ids(dataset_path, limit=3):
    def resolve_test_root(path):
        if os.path.isdir(path) and os.path.basename(os.path.normpath(path)) == "test":
            return path
        candidate = os.path.join(path, "test")
        if os.path.isdir(candidate):
            return candidate
        return path

    test_root = resolve_test_root(dataset_path)
    sat_dir = next((p for p in [os.path.join(test_root, "gallery_satellite"), os.path.join(test_root, "satellite")] if os.path.isdir(p)), None)
    drone_dir = next((p for p in [os.path.join(test_root, "query_drone"), os.path.join(test_root, "drone")] if os.path.isdir(p)), None)
    if sat_dir is None or drone_dir is None:
        raise ValueError(f"Could not find satellite/drone dirs under {test_root}")

    def is_dir_grouped(root_dir):
        return any(os.path.isdir(os.path.join(root_dir, name)) for name in os.listdir(root_dir))

    def ids_from_grouped(root_dir):
        return {name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))}

    def ids_from_flat(root_dir):
        exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        ids = set()
        for filename in os.listdir(root_dir):
            if filename.endswith(exts):
                ids.add(os.path.splitext(filename)[0].split("_")[0])
        return ids

    sat_ids = ids_from_grouped(sat_dir) if is_dir_grouped(sat_dir) else ids_from_flat(sat_dir)
    drone_ids = ids_from_grouped(drone_dir) if is_dir_grouped(drone_dir) else ids_from_flat(drone_dir)
    return sorted(list(sat_ids.intersection(drone_ids)))[:limit]


def create_comparison_figure(sat_original, drone_original, sat_backbone, drone_backbone, sat_vcsa, drone_vcsa, save_path):
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))

    axes[0, 0].imshow(sat_original)
    axes[0, 0].set_title("Satellite", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(drone_original)
    axes[0, 1].set_title("Drone", fontsize=14, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(sat_backbone)
    axes[1, 0].set_title("Backbone", fontsize=14, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(drone_backbone)
    axes[1, 1].axis("off")

    axes[2, 0].imshow(sat_vcsa)
    axes[2, 0].set_title("VCSA-Net", fontsize=14, fontweight="bold")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(drone_vcsa)
    axes[2, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate VCSA-Net heatmaps")
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--dataset_path", type=str, default="/root/dataset/University-Release")
    parser.add_argument("--sample_ids", type=str, nargs="+", default=None)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--output_dir", type=str, default="./heatmap_img_vcsa_net")
    parser.add_argument("--nclasses", type=int, default=701)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--triplet_loss", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_vrm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_sam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_csm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vcsa_residual_scale", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    class Config:
        def __init__(self):
            self.nclasses = args.nclasses
            self.block = args.block
            self.triplet_loss = args.triplet_loss
            self.resnet = False
            self.views = 2

    print("Loading model...")
    model = make_model(Config())
    if hasattr(model, "configure_vcsa_modules"):
        model.configure_vcsa_modules(
            use_vrm=args.use_vrm,
            use_sam=args.use_sam,
            use_csm=args.use_csm,
            residual_scale=args.vcsa_residual_scale,
        )

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt_path}")

    print(f"Loading checkpoint from: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=args.device)
    for key in [k for k in checkpoint.keys() if "classifier" in k and ("weight" in k or "bias" in k)]:
        del checkpoint[key]

    model.load_state_dict(checkpoint, strict=False)
    model = model.to(args.device)
    model.eval()

    model_with_vcsa = ModelWrapper(model, use_vcsa=True).eval()
    model_backbone = ModelWrapper(model, use_vcsa=False).eval()

    available_ids = list_available_sample_ids(args.dataset_path, limit=args.num_samples) if args.sample_ids is None else args.sample_ids
    if not available_ids:
        raise ValueError("No sample_ids found. Please pass --sample_ids explicitly.")

    print(f"Processing {len(available_ids)} samples...")
    for idx, sample_id in enumerate(available_ids):
        try:
            print(f"\nProcessing sample {sample_id} ({idx + 1}/{len(available_ids)})...")
            sat_tensor, drone_tensor, sat_original, drone_original = load_image_pair(args.dataset_path, sample_id, args.img_size)
            sat_tensor = sat_tensor.to(args.device)
            drone_tensor = drone_tensor.to(args.device)

            with torch.no_grad():
                sat_attn_vcsa, _ = model_with_vcsa(sat_tensor)
                drone_attn_vcsa, _ = model_with_vcsa(drone_tensor)
                sat_attn_backbone, _ = model_backbone(sat_tensor)
                drone_attn_backbone, _ = model_backbone(drone_tensor)

            sat_backbone, _ = generate_heatmap(sat_attn_backbone[0].cpu().numpy(), sat_original)
            drone_backbone, _ = generate_heatmap(drone_attn_backbone[0].cpu().numpy(), drone_original)
            sat_vcsa, _ = generate_heatmap(sat_attn_vcsa[0].cpu().numpy(), sat_original)
            drone_vcsa, _ = generate_heatmap(drone_attn_vcsa[0].cpu().numpy(), drone_original)

            save_path = os.path.join(args.output_dir, f"ablation_heatmap_{sample_id}.png")
            create_comparison_figure(sat_original, drone_original, sat_backbone, drone_backbone, sat_vcsa, drone_vcsa, save_path)
        except Exception as exc:
            print(f"Error processing sample {sample_id}: {exc}")

    print(f"\nAll heatmaps saved to {args.output_dir}")


if __name__ == "__main__":
    main()
