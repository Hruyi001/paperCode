#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate ablation heatmaps (Fig.4-style) for different attention modules.

We compute a similarity-driven Grad-CAM:
- pick matched (query, gallery) image pairs by shared folder id
- forward both through the model to get descriptors
- maximize cosine similarity between the pair (scalar objective)
- backprop to feature maps at the *attention output* and build Grad-CAM

Outputs a grid image with columns:
Input | baseline | CBAM | CA | ECA | EMA | GSRA
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
    gallery_tensor: torch.Tensor,
    model: GeoModel,
) -> np.ndarray:
    """
    Returns cam (H, W) in [0,1] for the *query* input, computed from attention output.
    """
    model.zero_grad(set_to_none=True)
    model.eval()

    q_desc = model(input_tensor)  # (1, D)
    g_desc = model(gallery_tensor)  # (1, D)
    sim = F.cosine_similarity(q_desc, g_desc, dim=1).mean()
    # maximize similarity -> take negative as loss to minimize
    loss = -sim
    loss.backward()

    if hook.activations is None or hook.gradients is None:
        raise RuntimeError("GradCAM hook did not capture activations/gradients. Check target layer.")

    # Grad-CAM: weights = GAP over spatial dims of gradients
    grads = hook.gradients  # (1, C, h, w)
    acts = hook.activations  # (1, C, h, w)
    weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = (weights * acts).sum(dim=1, keepdim=False)  # (1, h, w)
    cam = F.relu(cam)
    cam = cam[0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
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
    Query folders contain 1 image; gallery may contain multiple (we pick the first).
    """
    rng = np.random.default_rng(seed)
    q = get_data(query_root)
    g = get_data(gallery_root)
    ids = sorted(set(q.keys()).intersection(g.keys()))
    if len(ids) == 0:
        raise RuntimeError(
            "No overlapping IDs between query and gallery. "
            f"query_root={query_root} gallery_root={gallery_root}"
        )
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
    ap.add_argument("--data_root", type=str, required=True, help="Path to U1652 root folder")
    ap.add_argument("--num_pairs", type=int, default=3)
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

    # dataset roots
    if args.dataset == "U1652-D2S":
        query_root = os.path.join(args.data_root, "test", "query_drone")
        gallery_root = os.path.join(args.data_root, "test", "gallery_satellite")
    else:
        query_root = os.path.join(args.data_root, "test", "query_satellite")
        gallery_root = os.path.join(args.data_root, "test", "gallery_drone")

    variants: Dict[str, str] = json.loads(args.variant_checkpoints)
    # Only process available variants (skip missing ones)
    # Default order: Input first, then available variants in paper order
    # Paper order: baseline, CBAM, CA, ECA, EMA, GSRA
    paper_order = ["baseline", "CBAM", "CA", "ECA", "EMA", "GSRA"]
    available_variants = [v for v in paper_order if v in variants]
    # Add any other variants not in paper order
    for v in variants.keys():
        if v not in available_variants:
            available_variants.append(v)
    
    if not available_variants:
        raise ValueError("No variants provided in variant_checkpoints")
    col_order = ["Input"] + available_variants

    val_transforms, _, _ = get_transforms((args.img_size, args.img_size))

    pairs = pick_pairs(query_root, gallery_root, num_pairs=args.num_pairs, seed=args.seed)

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
    for name in col_order[1:]:  # Skip "Input"
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
    for q_path, g_path in pairs:
        q_rgb = load_image_rgb(q_path)
        g_rgb = load_image_rgb(g_path)

        q_t = val_transforms(image=q_rgb)["image"].unsqueeze(0).to(args.device)
        g_t = val_transforms(image=g_rgb)["image"].unsqueeze(0).to(args.device)

        # resize input image to match model input for display consistency
        q_show = cv2.resize(q_rgb, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
        row_imgs: List[np.ndarray] = [q_show]

        for name in col_order[1:]:  # Skip "Input"
            if name not in models:
                continue
            model = models[name]
            hook = hooks[name]
            cam = gradcam_from_hook(hook, q_t, g_t, model)
            cam = cv2.resize(cam, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
            row_imgs.append(overlay_cam(q_show, cam))

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

