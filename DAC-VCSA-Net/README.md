# VCSA-Net

VCSA-Net is a cross-view geo-localization method for University-1652 and SUES-200. The current implementation uses a hand ConvNeXt backbone with three alignment modules:

- **VRM**: Semantic-guided View Rectification Module
- **SAM**: Scale Adaptive Feature Calibration Module
- **CSM**: Cross-view Semantic Alignment Module

## 1. Environment

Use the conda environment prepared for this project:

```bash
conda activate <env-name>
```

All shell scripts run one process on one GPU. Use `GPU_ID` to select the GPU and `CONDA_ENV` to pass the environment name.

Default dataset paths:

```text
University-1652: /root/dataset/University-Release
SUES-200:        /root/dataset/SUES-200-512x512
```

## 2. Final Checkpoints

Only the final selected checkpoints are kept under `checkpoints/final/`:

```text
checkpoints/final/
├── university_d2s/weights_end.pth
├── university_s2d/weights_end.pth
├── sues_d2s_150/weights_end.pth
├── sues_d2s_200/weights_end.pth
├── sues_d2s_250/weights_end.pth
├── sues_d2s_300/weights_end.pth
├── sues_s2d_150/weights_end.pth
├── sues_s2d_200/weights_end.pth
├── sues_s2d_250/weights_end.pth
└── sues_s2d_300/weights_end.pth
```

Each final directory also contains `log.txt` for the corresponding evaluation record.

## 3. Inference / Evaluation Commands

### 3.1 University-1652 D→S

```bash
CONDA_ENV=<env-name> GPU_ID=0 DIRECTION=U1652-D2S bash scripts/eval_university.sh
```

Default checkpoint:

```text
checkpoints/final/university_d2s/weights_end.pth
```

### 3.2 University-1652 S→D

```bash
CONDA_ENV=<env-name> GPU_ID=0 DIRECTION=U1652-S2D bash scripts/eval_university.sh
```

Default checkpoint:

```text
checkpoints/final/university_s2d/weights_end.pth
```

### 3.3 SUES-200 D→S

```bash
CONDA_ENV=<env-name> GPU_ID=0 DIRECTION=U1652-D2S ALTITUDE=150 bash scripts/eval_sues200.sh
CONDA_ENV=<env-name> GPU_ID=0 DIRECTION=U1652-D2S ALTITUDE=200 bash scripts/eval_sues200.sh
CONDA_ENV=<env-name> GPU_ID=0 DIRECTION=U1652-D2S ALTITUDE=250 bash scripts/eval_sues200.sh
CONDA_ENV=<env-name> GPU_ID=0 DIRECTION=U1652-D2S ALTITUDE=300 bash scripts/eval_sues200.sh
```

Default checkpoints:

```text
checkpoints/final/sues_d2s_150/weights_end.pth
checkpoints/final/sues_d2s_200/weights_end.pth
checkpoints/final/sues_d2s_250/weights_end.pth
checkpoints/final/sues_d2s_300/weights_end.pth
```

### 3.4 SUES-200 S→D

```bash
CONDA_ENV=<env-name> GPU_ID=0 DIRECTION=U1652-S2D ALTITUDE=150 bash scripts/eval_sues200.sh
CONDA_ENV=<env-name> GPU_ID=0 DIRECTION=U1652-S2D ALTITUDE=200 bash scripts/eval_sues200.sh
CONDA_ENV=<env-name> GPU_ID=0 DIRECTION=U1652-S2D ALTITUDE=250 bash scripts/eval_sues200.sh
CONDA_ENV=<env-name> GPU_ID=0 DIRECTION=U1652-S2D ALTITUDE=300 bash scripts/eval_sues200.sh
```

Default checkpoints:

```text
checkpoints/final/sues_s2d_150/weights_end.pth
checkpoints/final/sues_s2d_200/weights_end.pth
checkpoints/final/sues_s2d_250/weights_end.pth
checkpoints/final/sues_s2d_300/weights_end.pth
```

You can still evaluate a custom checkpoint by passing `CKPT_PATH` explicitly:

```bash
CONDA_ENV=<env-name> GPU_ID=0 DIRECTION=U1652-D2S CKPT_PATH=/path/to/weights_end.pth bash scripts/eval_university.sh
```

## 4. Training Commands

The training scripts train new models with the current VCSA-Net code and hyperparameters. They do **not** automatically overwrite `checkpoints/final/`.

### 4.1 University-1652 D→S

```bash
CONDA_ENV=<env-name> GPU_ID=0 bash scripts/train_university_d2s.sh
```

Default output:

```text
checkpoints/university-vcsa-d2s/
```

### 4.2 University-1652 S→D

```bash
CONDA_ENV=<env-name> GPU_ID=1 bash scripts/train_university_s2d.sh
```

Default output:

```text
checkpoints/university-vcsa-s2d/
```

### 4.3 SUES-200 D→S

```bash
CONDA_ENV=<env-name> GPU_ID=2 DIRECTION=U1652-D2S ALTITUDE=200 bash scripts/train_sues200.sh
```

Default output:

```text
checkpoints/sues200-vcsa-U1652-D2S-200/
```

### 4.4 SUES-200 S→D

```bash
CONDA_ENV=<env-name> GPU_ID=3 DIRECTION=U1652-S2D ALTITUDE=200 bash scripts/train_sues200.sh
```

Default output:

```text
checkpoints/sues200-vcsa-U1652-S2D-200/
```

### 4.5 Continue Training from a Checkpoint

```bash
CONDA_ENV=<env-name> \
GPU_ID=0 \
DIRECTION=U1652-D2S \
ALTITUDE=150 \
CHECKPOINT_START=/path/to/weights_end.pth \
bash scripts/train_sues200.sh
```

If you want to replace a final checkpoint after retraining, manually copy the selected new `weights_end.pth` into the corresponding `checkpoints/final/.../weights_end.pth` directory after verifying the metrics.

## 5. Current Localization Accuracy

### 5.1 University-1652

| Direction | R@1 | R@5 | R@10 | R@top1 | AP |
|---|---:|---:|---:|---:|---:|
| D→S | 92.9121 | 97.5352 | 98.2775 | 98.3304 | 93.9698 |
| S→D | 96.4337 | 97.8602 | 97.8602 | 99.5720 | 93.7961 |

### 5.2 SUES-200 D→S

| Altitude | R@1 | R@5 | R@10 | R@top1 | AP |
|---|---:|---:|---:|---:|---:|
| 150m | 91.8000 | 99.2500 | 99.8250 | 98.1750 | 93.4062 |
| 200m | 92.9500 | 99.8000 | 99.9500 | 99.2750 | 94.4436 |
| 250m | 96.1750 | 99.8250 | 100.0000 | 99.2750 | 96.9879 |
| 300m | 95.4750 | 99.7250 | 100.0000 | 98.9250 | 96.3992 |

### 5.3 SUES-200 S→D

| Altitude | R@1 | R@5 | R@10 | R@top1 | AP |
|---|---:|---:|---:|---:|---:|
| 150m | 96.2500 | 98.7500 | 98.7500 | 100.0000 | 93.1845 |
| 200m | 96.2500 | 98.7500 | 100.0000 | 100.0000 | 93.5720 |
| 250m | 97.5000 | 98.7500 | 98.7500 | 100.0000 | 96.7326 |
| 300m | 97.5000 | 98.7500 | 98.7500 | 100.0000 | 97.0963 |

## 6. Heatmap Generation

Heatmap code is under `heatmap_code_v1/`:

```bash
CONDA_ENV=<env-name> GPU_ID=0 bash heatmap_code_v1/run_generate_heatmap.sh
```

Default output directory:

```text
heatmap_code_v1/heatmap_img_vcsa_net
```

Heatmap layout:

- Row 1: satellite / drone original images
- Row 2: backbone heatmaps
- Row 3: VCSA-Net heatmaps

## 7. Notes

- `checkpoints/final/` stores the current selected final models.
- Training scripts produce new candidate models and save them outside `checkpoints/final/` by default.
- To run multiple jobs, start multiple single-GPU commands with different `GPU_ID` values.
- For custom evaluation, pass `CKPT_PATH=/path/to/weights_end.pth`.
