# VCSA-Net Heatmap Code

This folder contains the heatmap generation script for the current VCSA-Net model.

## Files

- `generate_heatmap.py` - heatmap generation script
- `run_generate_heatmap.sh` - shell entrypoint
- `heatmap_img_vcsa_net/` - generated example heatmaps

## Usage

```bash
CONDA_ENV=<training-env> GPU_ID=0 bash heatmap_code_v1/run_generate_heatmap.sh
```

Optional overrides:

```bash
CONDA_ENV=<training-env> \
GPU_ID=0 \
CKPT_PATH=/path/to/weights_end.pth \
DATASET_PATH=/root/dataset/University-Release \
OUTPUT_DIR=heatmap_code_v1/heatmap_img_vcsa_net \
bash heatmap_code_v1/run_generate_heatmap.sh
```

## Visualization Layout

- Row 1: satellite / drone original images
- Row 2: backbone heatmaps
- Row 3: VCSA-Net heatmaps generated from VRM/SAM/CSM-enhanced part features
