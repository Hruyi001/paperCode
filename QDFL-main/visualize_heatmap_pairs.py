#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
成对可视化（无人机-卫星）热力图结果：
- 第一列：配对的输入图（上：Drone，下：Satellite）
- 第二列：Overlay(Backbone featmap)（上/下分别对应 Drone/Satellite）
- 第三列：Overlay(QDFL x_fine_0)（上/下分别对应 Drone/Satellite）

配对方式：
- by_relpath：以各自目录下“相对路径（去掉扩展名）”作为 key 取交集一一配对。
- by_id：以“第一级目录（如 0001）”作为地点/ID。每个 ID 从 satellite_dir 中选 1 张卫星图，
  然后与该 ID 下所有无人机图逐一配对（适合 University-1652 常见的 1 张卫星对多张无人机）。
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.evaluation_utils.load_network import load_network_supervised
from utils.commons import load_config
from visualize_heatmap import HeatmapVisualizer


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def list_images_recursive(root_dir: str):
    root = Path(root_dir)
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return sorted(files)


def build_key_map(root_dir: str):
    root = Path(root_dir).resolve()
    m = {}
    for p in list_images_recursive(str(root)):
        rel = p.resolve().relative_to(root)
        key = str(rel.with_suffix("")).replace("\\", "/")
        m[key] = p
    return m


def build_id_groups(root_dir: str):
    """
    按第一级目录分组：
      root/0001/xxx.jpg -> id='0001'
    对于不在子目录下的图片，会归到 id=''。
    """
    root = Path(root_dir).resolve()
    groups = {}
    for p in list_images_recursive(str(root)):
        rel = p.resolve().relative_to(root)
        parts = rel.parts
        id_ = parts[0] if len(parts) >= 2 else ""
        groups.setdefault(id_, []).append(p)
    # sort for determinism
    for k in groups:
        groups[k] = sorted(groups[k])
    return groups


def overlay_on_image(img_array: np.ndarray, heatmap_0_1: np.ndarray, alpha: float, cmap_name: str):
    """
    img_array: uint8 [H,W,3]
    heatmap_0_1: float [H,W] in 0..1
    """
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(heatmap_0_1)  # RGBA float 0..1
    colored_rgb = (colored[:, :, :3] * 255.0).astype(np.uint8)
    base = img_array.astype(np.float32)
    over = (1.0 - alpha) * base + alpha * colored_rgb.astype(np.float32)
    return np.clip(over, 0, 255).astype(np.uint8)


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Paired (Drone/Satellite) heatmap overlay visualization")
    parser.add_argument("--drone_dir", type=str, required=True, help="Drone images root dir (recursive)")
    parser.add_argument("--satellite_dir", type=str, required=True, help="Satellite images root dir (recursive)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--config", type=str, default="./model_configs/dino_b_QDFL.yaml", help="Path to model config YAML")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_pairs", type=int, default=0, help="How many pairs to export (0 = all)")
    parser.add_argument(
        "--pair_mode",
        type=str,
        default="by_id",
        choices=["by_id", "by_relpath"],
        help="Pairing mode: by_id (one satellite per ID, many drones) or by_relpath (1-1 by relative path stem)",
    )
    parser.add_argument(
        "--satellite_pick",
        type=str,
        default="first",
        choices=["first", "last"],
        help="When pair_mode=by_id, which satellite image to pick within each ID folder",
    )
    parser.add_argument("--img_size", type=int, nargs=2, default=[280, 280], help="Input image size [height width]")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha in [0,1]")
    parser.add_argument("--heatmap_method", type=str, default="mean", choices=["mean", "max", "norm"])
    parser.add_argument("--cmap", type=str, default="jet", help="Matplotlib colormap name")
    args = parser.parse_args()

    if not os.path.isdir(args.drone_dir):
        raise FileNotFoundError(f"drone_dir does not exist: {args.drone_dir}")
    if not os.path.isdir(args.satellite_dir):
        raise FileNotFoundError(f"satellite_dir does not exist: {args.satellite_dir}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"config does not exist: {args.config}")

    os.makedirs(args.output_dir, exist_ok=True)

    # load model
    config = load_config(args.config)
    model_configs = config["model_configs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_network_supervised(model_configs, args.checkpoint)
    model.eval()

    visualizer = HeatmapVisualizer(model, device, img_size=tuple(args.img_size))

    # pairing
    out_root = Path(args.output_dir).resolve()
    pairs = []

    if args.pair_mode == "by_relpath":
        drone_map = build_key_map(args.drone_dir)
        sat_map = build_key_map(args.satellite_dir)
        keys = sorted(set(drone_map.keys()) & set(sat_map.keys()))
        if len(keys) == 0:
            raise RuntimeError("No paired images found (intersection of relative-path keys is empty).")
        for key in keys:
            pairs.append((key, drone_map[key], sat_map[key]))
    else:
        drone_groups = build_id_groups(args.drone_dir)
        sat_groups = build_id_groups(args.satellite_dir)
        ids = sorted(set(drone_groups.keys()) & set(sat_groups.keys()))
        if len(ids) == 0:
            raise RuntimeError("No paired IDs found (intersection of first-level folder names is empty).")
        for id_ in ids:
            sat_list = sat_groups.get(id_, [])
            drone_list = drone_groups.get(id_, [])
            if len(sat_list) == 0 or len(drone_list) == 0:
                continue
            sat_pick = sat_list[0] if args.satellite_pick == "first" else sat_list[-1]
            for d in drone_list:
                key = f"{id_}/{d.stem}"
                pairs.append((key, d, sat_pick))

    if args.num_pairs and args.num_pairs > 0:
        pairs = pairs[: args.num_pairs]

    print(f"Found {len(pairs)} pairs (mode={args.pair_mode}). Exporting to: {out_root}")

    for idx, (key, drone_p, sat_p) in enumerate(pairs, start=1):
        drone_path = str(drone_p)
        sat_path = str(sat_p)

        d_img, d_qdfl, d_backbone = visualizer.extract_heatmaps(drone_path, method=args.heatmap_method)
        s_img, s_qdfl, s_backbone = visualizer.extract_heatmaps(sat_path, method=args.heatmap_method)
        if d_img is None or s_img is None:
            print(f"[{idx}/{len(pairs)}] Skip (failed extraction): {key}")
            continue

        # backbone heatmap 可能为 None（依赖 backbone 输出结构）
        if d_backbone is None or s_backbone is None:
            print(f"[{idx}/{len(pairs)}] Skip (missing backbone featmap): {key}")
            continue

        d_over_backbone = overlay_on_image(d_img, d_backbone, alpha=args.alpha, cmap_name=args.cmap)
        s_over_backbone = overlay_on_image(s_img, s_backbone, alpha=args.alpha, cmap_name=args.cmap)
        d_over_qdfl = overlay_on_image(d_img, d_qdfl, alpha=args.alpha, cmap_name=args.cmap)
        s_over_qdfl = overlay_on_image(s_img, s_qdfl, alpha=args.alpha, cmap_name=args.cmap)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        # column 1: inputs
        axes[0, 0].imshow(d_img)
        axes[0, 0].set_title("Input (Drone)", fontsize=12)
        axes[0, 0].axis("off")
        axes[1, 0].imshow(s_img)
        axes[1, 0].set_title("Input (Satellite)", fontsize=12)
        axes[1, 0].axis("off")

        # column 2: backbone overlay
        axes[0, 1].imshow(d_over_backbone)
        axes[0, 1].set_title("Overlay (Backbone featmap) - Drone", fontsize=12)
        axes[0, 1].axis("off")
        axes[1, 1].imshow(s_over_backbone)
        axes[1, 1].set_title("Overlay (Backbone featmap) - Satellite", fontsize=12)
        axes[1, 1].axis("off")

        # column 3: qdfl overlay
        axes[0, 2].imshow(d_over_qdfl)
        axes[0, 2].set_title("Overlay (QDFL x_fine_0) - Drone", fontsize=12)
        axes[0, 2].axis("off")
        axes[1, 2].imshow(s_over_qdfl)
        axes[1, 2].set_title("Overlay (QDFL x_fine_0) - Satellite", fontsize=12)
        axes[1, 2].axis("off")

        # add key on figure for traceability
        fig.suptitle(key, fontsize=10)
        plt.tight_layout()

        out_path = out_root / f"{key}_pair.png"
        ensure_parent_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if idx % 25 == 0 or idx == len(pairs):
            print(f"[{idx}/{len(pairs)}] {key} -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()

