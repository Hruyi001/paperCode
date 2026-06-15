#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单对（无人机/卫星）可视化：
- 第一列：输入图（上 Drone / 下 Satellite）
- 第二列：Overlay(Backbone featmap)
- 第三列：Overlay(QDFL x_fine_0)

用户手动指定两张图的路径，不做任何自动配对。
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.cm as cm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.evaluation_utils.load_network import load_network_supervised
from utils.commons import load_config
from visualize_heatmap import HeatmapVisualizer


def overlay_on_image(img_array: np.ndarray, heatmap_0_1: np.ndarray, alpha: float, cmap_name: str):
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(heatmap_0_1)  # RGBA float 0..1
    colored_rgb = (colored[:, :, :3] * 255.0).astype(np.uint8)
    base = img_array.astype(np.float32)
    over = (1.0 - alpha) * base + alpha * colored_rgb.astype(np.float32)
    return np.clip(over, 0, 255).astype(np.uint8)


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    # keep it filesystem-friendly
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", ".", "+"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "heatmap_pair"


def main():
    parser = argparse.ArgumentParser(description="Visualize a single Drone/Satellite pair and export 4 separate overlay images")
    parser.add_argument("--drone_image", type=str, required=True, help="Path to a drone image")
    parser.add_argument("--satellite_image", type=str, required=True, help="Path to a satellite image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--config", type=str, default="./model_configs/dino_b_QDFL.yaml", help="Path to model config YAML")
    parser.add_argument("--output_path", type=str, default="", help="(Optional) Base path; directory and stem used as output_dir/prefix.")
    parser.add_argument("--output_dir", type=str, default="./heatmap_visualizations_pairs_single", help="Directory to save result images")
    parser.add_argument("--output_name", type=str, default="", help="Prefix for output filenames (without extension). Default auto-generated from input names.")
    parser.add_argument("--img_size", type=int, nargs=2, default=[280, 280], help="Input image size [height width]")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha in [0,1]")
    parser.add_argument("--heatmap_method", type=str, default="mean", choices=["mean", "max", "norm"])
    parser.add_argument("--cmap", type=str, default="jet", help="Matplotlib colormap name")
    parser.add_argument("--title", type=str, default="", help="Optional figure title")
    args = parser.parse_args()

    if not os.path.isfile(args.drone_image):
        raise FileNotFoundError(f"drone_image does not exist: {args.drone_image}")
    if not os.path.isfile(args.satellite_image):
        raise FileNotFoundError(f"satellite_image does not exist: {args.satellite_image}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"config does not exist: {args.config}")

    # 决定输出目录和前缀
    if args.output_path:
        base = Path(args.output_path)
        out_dir = base.parent
        prefix = base.stem
    else:
        out_dir = Path(args.output_dir)
        d = Path(args.drone_image)
        s = Path(args.satellite_image)
        default_name = f"d-{d.stem}__s-{s.stem}__a{args.alpha:g}__m-{args.heatmap_method}"
        prefix = args.output_name.strip() or default_name
    prefix = sanitize_filename(prefix)
    ensure_parent_dir(out_dir / "dummy")  # 确保目录存在（占位文件名会被忽略）

    # load model
    config = load_config(args.config)
    model_configs = config["model_configs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_network_supervised(model_configs, args.checkpoint)
    model.eval()

    visualizer = HeatmapVisualizer(model, device, img_size=tuple(args.img_size))

    d_img, d_qdfl, d_backbone = visualizer.extract_heatmaps(args.drone_image, method=args.heatmap_method)
    s_img, s_qdfl, s_backbone = visualizer.extract_heatmaps(args.satellite_image, method=args.heatmap_method)

    if d_img is None or s_img is None or d_qdfl is None or s_qdfl is None:
        raise RuntimeError("Failed to extract QDFL heatmaps for one/both images.")
    if d_backbone is None or s_backbone is None:
        raise RuntimeError("Failed to extract backbone heatmaps for one/both images (backbone output may be incompatible).")

    d_over_backbone = overlay_on_image(d_img, d_backbone, alpha=args.alpha, cmap_name=args.cmap)
    s_over_backbone = overlay_on_image(s_img, s_backbone, alpha=args.alpha, cmap_name=args.cmap)
    d_over_qdfl = overlay_on_image(d_img, d_qdfl, alpha=args.alpha, cmap_name=args.cmap)
    s_over_qdfl = overlay_on_image(s_img, s_qdfl, alpha=args.alpha, cmap_name=args.cmap)

    # 分别保存四张 overlay 图
    out_drone_backbone = out_dir / f"{prefix}_drone_backbone.png"
    out_sat_backbone = out_dir / f"{prefix}_satellite_backbone.png"
    out_drone_qdfl = out_dir / f"{prefix}_drone_qdfl.png"
    out_sat_qdfl = out_dir / f"{prefix}_satellite_qdfl.png"

    Image.fromarray(d_over_backbone).save(out_drone_backbone)
    Image.fromarray(s_over_backbone).save(out_sat_backbone)
    Image.fromarray(d_over_qdfl).save(out_drone_qdfl)
    Image.fromarray(s_over_qdfl).save(out_sat_qdfl)

    print("Saved:")
    print(f"  {out_drone_backbone}")
    print(f"  {out_sat_backbone}")
    print(f"  {out_drone_qdfl}")
    print(f"  {out_sat_qdfl}")


if __name__ == "__main__":
    main()

