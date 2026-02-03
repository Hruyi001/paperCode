#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量生成 QDFL x_fine_0 的 overlay 热力图：
- 输入：一个目录（递归扫描所有图片）
- 输出：把 overlay 结果保存到输出目录（默认保留相对目录结构）
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.cm as cm
from torchvision import transforms

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.evaluation_utils.load_network import load_network_supervised
from utils.commons import load_config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def list_images_recursive(root_dir: str):
    root = Path(root_dir)
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return sorted(files)


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_0_1(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    return (x - mn) / (mx - mn + 1e-8)


class QDFLOverlayExporter:
    def __init__(self, model, device, img_size=(280, 280)):
        self.model = model
        self.device = device
        self.img_size = img_size
        self.x_fine_0 = None

        self._register_hook()

        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _register_hook(self):
        """注册hook来捕获AQEU模块输出的x_fine_0特征图"""

        def hook_fn(module, inputs, output):
            if isinstance(output, tuple) and len(output) == 2:
                self.x_fine_0 = output[1].detach().cpu()

        # 找到AQEU模块并注册hook
        for name, module in self.model.named_modules():
            if "AQEU" in name and hasattr(module, "forward"):
                module.register_forward_hook(hook_fn)
                return

        # 兜底：常见结构
        if hasattr(self.model, "components") and hasattr(self.model.components, "AQEU"):
            self.model.components.AQEU.register_forward_hook(hook_fn)

    def _generate_heatmap_2d(self, feature_map: torch.Tensor, method: str) -> np.ndarray:
        """
        feature_map: [1, C, H, W] or [C, H, W]
        """
        fm = feature_map
        if isinstance(fm, torch.Tensor):
            fm = fm.numpy()
        if fm.ndim == 4:
            fm = fm[0]  # [C,H,W]
        if method == "mean":
            h2d = np.mean(fm, axis=0)
        elif method == "max":
            h2d = np.max(fm, axis=0)
        elif method == "norm":
            h2d = np.linalg.norm(fm, axis=0)
        else:
            h2d = np.mean(fm, axis=0)
        return normalize_0_1(h2d)

    def export_overlay(self, img_path: Path, out_path: Path, alpha: float, method: str, cmap_name: str):
        # load image
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        self.x_fine_0 = None
        with torch.no_grad():
            _ = self.model(img_tensor)

        if self.x_fine_0 is None:
            raise RuntimeError("Failed to capture x_fine_0 (AQEU hook not triggered).")

        heatmap = self._generate_heatmap_2d(self.x_fine_0, method=method)  # [H,W] 0..1
        heatmap_resized = np.array(Image.fromarray(heatmap).resize(img.size, Image.BICUBIC))  # [H,W]

        # colormap -> RGBA
        cmap = cm.get_cmap(cmap_name)
        colored = cmap(heatmap_resized)  # float RGBA in 0..1
        colored_rgb = (colored[:, :, :3] * 255.0).astype(np.uint8)

        base = np.array(img).astype(np.float32)
        overlay = (1.0 - alpha) * base + alpha * colored_rgb.astype(np.float32)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        ensure_parent_dir(out_path)
        Image.fromarray(overlay).save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Batch export QDFL x_fine_0 overlay heatmaps for all images in a directory")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing images (recursive)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save overlays")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--config", type=str, default="./model_configs/dino_b_QDFL.yaml", help="Path to model config YAML")
    parser.add_argument("--img_size", type=int, nargs=2, default=[280, 280], help="Input image size [height width]")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha in [0,1]")
    parser.add_argument("--heatmap_method", type=str, default="mean", choices=["mean", "max", "norm"])
    parser.add_argument("--cmap", type=str, default="jet", help="Matplotlib colormap name (default: jet)")
    parser.add_argument("--preserve_structure", action="store_true", help="Preserve relative folder structure under output_dir")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"input_dir does not exist: {args.input_dir}")
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

    exporter = QDFLOverlayExporter(model, device, img_size=tuple(args.img_size))

    images = list_images_recursive(args.input_dir)
    if len(images) == 0:
        print(f"No images found in {args.input_dir}")
        return

    in_root = Path(args.input_dir).resolve()
    out_root = Path(args.output_dir).resolve()

    print(f"Found {len(images)} images. Exporting overlays...")
    for i, img_path in enumerate(images, start=1):
        rel = img_path.resolve().relative_to(in_root)
        if args.preserve_structure:
            out_path = out_root / rel
        else:
            out_path = out_root / rel.name
        # ensure png output for consistency
        out_path = out_path.with_suffix(".png")
        try:
            exporter.export_overlay(img_path, out_path, alpha=args.alpha, method=args.heatmap_method, cmap_name=args.cmap)
            if i % 50 == 0 or i == len(images):
                print(f"[{i}/{len(images)}] {img_path.name} -> {out_path}")
        except Exception as e:
            print(f"Failed on {img_path}: {e}")

    print(f"Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

