#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate GSRA heatmaps only, overlaid on original images.

For each image in the input directory:
- forward through the model to get descriptor
- maximize feature magnitude (L2 norm) as objective
- backprop to feature maps at the *attention output* and build Grad-CAM

Outputs a grid image with GSRA heatmaps overlaid on original images.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datasets.university import get_data, get_transforms
from models.model import GeoModel


@dataclass
class ModelCfg:
    backbone: str
    attention: Optional[str]
    aggregation: str
    num_channels: int
    img_size: int
    num_clusters: int
    cluster_dim: int
    device: str


class GradCAMHook:
    """Capture feature map activations + gradients from a target module output."""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle = module.register_forward_hook(self._forward_hook)
        self._bwd_handle = module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output):
        # output: (B, C, H, W)
        self.activations = output

    def _backward_hook(self, _module, _grad_input, grad_output):
        # grad_output[0]: gradient w.r.t module output
        self.gradients = grad_output[0]

    def close(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def _ensure_rgb_uint8(img_rgb: np.ndarray) -> np.ndarray:
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    return img_rgb


def overlay_cam(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    img_rgb: HxWx3 uint8
    cam: HxW float in [0,1]
    """
    img_rgb = _ensure_rgb_uint8(img_rgb)
    cam_u8 = np.clip(cam * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)  # BGR
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (img_rgb.astype(np.float32) * (1.0 - alpha)) + (heat.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def gradcam_from_hook(
    hook: GradCAMHook,
    input_tensor: torch.Tensor,
    gallery_tensor: Optional[torch.Tensor] = None,
    model: Optional[GeoModel] = None,
) -> np.ndarray:
    """
    Returns cam (H, W) in [0,1] for the input, computed from attention output.
    If gallery_tensor is provided, uses similarity-driven Grad-CAM.
    Otherwise, uses feature magnitude-driven Grad-CAM for single image.
    """
    if model is None:
        raise ValueError("Model must be provided")
    
    model.zero_grad(set_to_none=True)
    # Need to enable gradient computation
    input_tensor.requires_grad_(True)
    
    # Set model to eval mode but enable gradients for parameters
    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    if gallery_tensor is not None:
        # Pair-based: similarity-driven Grad-CAM
        q_desc = model(input_tensor)  # (1, D)
        g_desc = model(gallery_tensor)  # (1, D)
        sim = F.cosine_similarity(q_desc, g_desc, dim=1).mean()
        # maximize similarity -> take negative as loss to minimize
        loss = -sim
    else:
        # Single image: use feature magnitude as target
        # Forward through model to get descriptor
        desc = model(input_tensor)  # (1, D)
        # Maximize the L2 norm of the descriptor (use sum for stronger signal)
        loss = -desc.norm(dim=1).sum()
    
    loss.backward()

    if hook.activations is None:
        raise RuntimeError("GradCAM hook did not capture activations. Check target layer.")
    
    acts = hook.activations  # (1, C, h, w)
    
    if hook.gradients is None:
        print("Warning: Gradients are None, using activation-based CAM instead of Grad-CAM")
        # Fallback to activation-based CAM
        cam = acts.mean(dim=1, keepdim=False)  # (1, h, w)
        cam = F.relu(cam)
        cam = cam[0]
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max < 1e-6:
            print(f"Warning: CAM values are very small (max={cam_max})")
        cam = cam / (cam_max + 1e-6)
        return cam.detach().cpu().numpy().astype(np.float32)

    # Grad-CAM: weights = GAP over spatial dims of gradients
    grads = hook.gradients  # (1, C, h, w)
    
    # Check if gradients are all zero or very small
    grad_magnitude = grads.abs().sum().item()
    if grad_magnitude < 1e-8:
        print(f"Warning: Gradients are very small (magnitude={grad_magnitude}), using activation-based CAM")
        # Fallback to activation-based CAM
        cam = acts.mean(dim=1, keepdim=False)  # (1, h, w)
    else:
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=False)  # (1, h, w)
    
    cam = F.relu(cam)
    cam = cam[0]
    cam = cam - cam.min()
    cam_max = cam.max()
    if cam_max < 1e-6:
        print(f"Warning: CAM values are very small (max={cam_max}), heatmap might appear empty")
    cam = cam / (cam_max + 1e-6)
    return cam.detach().cpu().numpy().astype(np.float32)


def load_image_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def pick_pairs(
    query_root: str,
    gallery_root: str,
    num_pairs: int,
    seed: int = 1,
) -> List[Tuple[str, str]]:
    """
    Picks (query_img_path, gallery_img_path) with the same folder-id.
    Supports both directory-based structure and flat file structure.
    For flat structure: query images are {id}.jpg, gallery images are {id}-0.png
    """
    rng = np.random.default_rng(seed)
    
    # Try directory-based structure first
    q = get_data(query_root)
    g = get_data(gallery_root)
    ids = sorted(set(q.keys()).intersection(g.keys()))
    
    # If no directory-based structure found, try flat file structure
    if len(ids) == 0:
        # Check if query_root and gallery_root are the same directory (flat structure)
        # Normalize paths to handle different representations of the same path
        query_dir = os.path.abspath(query_root) if os.path.isdir(query_root) else os.path.abspath(os.path.dirname(query_root))
        gallery_dir = os.path.abspath(gallery_root) if os.path.isdir(gallery_root) else os.path.abspath(os.path.dirname(gallery_root))
        
        if query_root == gallery_root or query_dir == gallery_dir:
            # Flat structure: all images in one directory
            img_dir = query_root if os.path.isdir(query_root) else os.path.dirname(query_root)
            if not os.path.isdir(img_dir):
                img_dir = gallery_root if os.path.isdir(gallery_root) else os.path.dirname(gallery_root)
            
            # Find all .jpg files (query) and -0.png files (gallery)
            import glob
            query_files = {}
            gallery_files = {}
            
            for jpg_file in glob.glob(os.path.join(img_dir, "*.jpg")):
                basename = os.path.basename(jpg_file)
                img_id = os.path.splitext(basename)[0]
                query_files[img_id] = jpg_file
            
            for png_file in glob.glob(os.path.join(img_dir, "*-0.png")):
                basename = os.path.basename(png_file)
                # Extract ID from filename like "26-0.png" -> "26"
                img_id = basename.split("-")[0]
                gallery_files[img_id] = png_file
            
            ids = sorted(set(query_files.keys()).intersection(gallery_files.keys()))
            
            if len(ids) == 0:
                raise RuntimeError(
                    "No overlapping IDs between query and gallery. "
                    f"query_root={query_root} gallery_root={gallery_root}\n"
                    f"Found {len(query_files)} query files and {len(gallery_files)} gallery files."
                )
            
            if num_pairs > len(ids):
                num_pairs = len(ids)
            chosen = rng.choice(ids, size=num_pairs, replace=False).tolist()
            
            pairs: List[Tuple[str, str]] = []
            for sid in chosen:
                pairs.append((query_files[sid], gallery_files[sid]))
            return pairs
        else:
            raise RuntimeError(
                "No overlapping IDs between query and gallery. "
                f"query_root={query_root} gallery_root={gallery_root}"
            )
    
    # Directory-based structure
    if num_pairs > len(ids):
        num_pairs = len(ids)
    chosen = rng.choice(ids, size=num_pairs, replace=False).tolist()

    pairs: List[Tuple[str, str]] = []
    for sid in chosen:
        q_path = os.path.join(q[sid]["path"], q[sid]["files"][0])
        g_path = os.path.join(g[sid]["path"], g[sid]["files"][0])
        pairs.append((q_path, g_path))
    return pairs


def build_model(cfg: ModelCfg, checkpoint_path: str) -> GeoModel:
    model_cfg = type("Configuration", (), {})()
    model_cfg.backbone = cfg.backbone
    model_cfg.attention = cfg.attention
    model_cfg.aggregation = cfg.aggregation
    model_cfg.num_channels = cfg.num_channels
    model_cfg.img_size = cfg.img_size
    model_cfg.num_clusters = cfg.num_clusters
    model_cfg.cluster_dim = cfg.cluster_dim

    model = GeoModel(model_cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(cfg.device)
    model.eval()
    return model


def make_grid(
    rows: List[List[np.ndarray]],
    col_titles: List[str],
    pad: int = 8,
    title_h: int = 26,
) -> np.ndarray:
    """
    rows: list of rows; each row is list of RGB uint8 images with same size.
    """
    if len(rows) == 0 or len(rows[0]) == 0:
        raise ValueError("Empty rows for grid.")
    h, w, _ = rows[0][0].shape
    nrows = len(rows)
    ncols = len(rows[0])
    assert all(len(r) == ncols for r in rows)

    grid_h = title_h + nrows * h + (nrows + 1) * pad
    grid_w = ncols * w + (ncols + 1) * pad
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    # titles
    for c, t in enumerate(col_titles):
        x0 = pad + c * (w + pad)
        cv2.putText(
            canvas,
            t,
            (x0 + 2, int(title_h * 0.75)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # images
    y = title_h + pad
    for r in range(nrows):
        x = pad
        for c in range(ncols):
            canvas[y : y + h, x : x + w] = rows[r][c]
            x += w + pad
        y += h + pad
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, choices=["U1652-D2S", "U1652-S2D"], default="U1652-S2D")
    ap.add_argument("--data_root", type=str, required=True, help="Path to directory containing input images")
    ap.add_argument("--num_pairs", type=int, default=3, help="Number of images to process (0 for all)")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--variant_checkpoints",
        type=str,
        required=True,
        help="JSON: {\"baseline\": \"/path/best.pth\", \"GSRA\": \"/path/best.pth\", ...}",
    )
    ap.add_argument("--out_dir", type=str, default="heatpmap_img")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    variants: Dict[str, str] = json.loads(args.variant_checkpoints)
    # Only process GSRA variant
    if "GSRA" not in variants:
        raise ValueError("GSRA checkpoint not found in variant_checkpoints")
    col_order = ["GSRA"]

    val_transforms, _, _ = get_transforms((args.img_size, args.img_size))

    # Get all image files from data_root directory
    import glob
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.data_root, ext)))
    
    if len(image_files) == 0:
        raise RuntimeError(f"No image files found in {args.data_root}")
    
    # Limit number of images if specified
    if args.num_pairs > 0 and args.num_pairs < len(image_files):
        rng = np.random.default_rng(args.seed)
        image_files = rng.choice(image_files, size=args.num_pairs, replace=False).tolist()
    else:
        image_files = image_files[:args.num_pairs] if args.num_pairs > 0 else image_files

    # Preload models per variant
    # Note: backbone output channels in this repo are 768 (ConvNeXt tiny cropped) by default,
    # but configs in training use num_channels=384. We stick to config used by checkpoints.
    # If your checkpoints were trained with different configs, pass matching values by editing below.
    base_cfg = ModelCfg(
        backbone="ConvNeXt",
        attention="GSRA",
        aggregation="SALAD",
        num_channels=384,
        img_size=args.img_size,
        num_clusters=128,
        cluster_dim=64,
        device=args.device,
    )

    models: Dict[str, GeoModel] = {}
    hooks: Dict[str, GradCAMHook] = {}
    for name in col_order:
        if name not in variants:
            continue
        attn = None if name == "baseline" else name
        cfg = ModelCfg(**{**base_cfg.__dict__, "attention": attn})
        model = build_model(cfg, variants[name])
        # hook attention output for CAM
        hook = GradCAMHook(model.model.attention)
        models[name] = model
        hooks[name] = hook

    rows: List[List[np.ndarray]] = []
    for img_path in image_files:
        img_rgb = load_image_rgb(img_path)

        img_t = val_transforms(image=img_rgb)["image"].unsqueeze(0).to(args.device)

        # Resize input image to match model input for display consistency
        img_show = cv2.resize(img_rgb, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
        # Only process GSRA heatmap, overlay on original image
        row_imgs: List[np.ndarray] = []

        for name in col_order:
            if name not in models:
                continue
            model = models[name]
            hook = hooks[name]
            # Generate heatmap for single image (no pairing)
            cam = gradcam_from_hook(hook, img_t, gallery_tensor=None, model=model)
            cam = cv2.resize(cam, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
            # Overlay heatmap on original image
            row_imgs.append(overlay_cam(img_show, cam))

        rows.append(row_imgs)

    # cleanup hooks
    for h in hooks.values():
        h.close()

    grid = make_grid(rows, col_titles=col_order)
    out_path = os.path.join(args.out_dir, f"heatmap_ablation_{args.dataset}.png")
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()

