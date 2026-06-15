# FCFE-Net

FCFE-Net is a cross-view geo-localization method for UAV-oriented image retrieval. The current implementation builds on a ConvNeXt retrieval backbone and adds three Chapter 4 feature enhancement modules:

- **MGSE**: multi-granularity semantic-geometric enhancement using RGB features and Omnidata normal features.
- **SEM**: semantic enhancement with spatial and channel attention.
- **FSM**: frequency/background suppression for more discriminative retrieval descriptors.

The FCFE path is optional in the code, but the recommended current method uses `--use_fcfe --use_normals` with cached Omnidata normal images.

## Environment

Use the `ccr` conda environment for training, testing, and verification:

```bash
conda run -n ccr python -m py_compile train_university.py test_university.py utils.py
```

Install Python dependencies if needed:

```bash
pip install -r requirement.txt
```

Omnidata normal generation expects the Omnidata repository and pretrained models at:

```text
/root/code/omnidata_models
/root/code/omnidata_models/pretrained_models
```

## Dataset layout

### University-1652

Expected structure:

```text
University-Release/
├── train/
│   ├── satellite/
│   ├── drone/
│   ├── street/
│   └── google/
└── test/
    ├── query_drone/
    ├── gallery_drone/
    ├── query_street/
    ├── gallery_street/
    ├── query_satellite/
    ├── gallery_satellite/
    └── 4K_drone/
```

The local paths used in this environment are:

```text
/root/dataset/University-Release/train
/root/dataset/University-Release/test
```

### SUES-200

Expected structure for direct training:

```text
SUES-200-512x512/
├── Training/
│   └── 200/
│       ├── satellite/
│       └── drone/
└── Testing/
    └── 200/
        ├── query_satellite/
        ├── gallery_satellite/
        ├── query_drone/
        └── gallery_drone/
```

The local root used in this environment is:

```text
/root/dataset/SUES-200-512x512
```

## Omnidata normal cache

FCFE-Net uses surface normal images as geometric priors. Normals are generated from RGB images with Omnidata and cached on disk.

Recommended cache directories:

```text
/root/dataset/University-Release/omnidata_normals_train
/root/dataset/University-Release/omnidata_normals_test
```

The training and testing scripts can generate missing normals when `--auto_generate_normals` is enabled, but for faster experiments it is better to precompute or reuse existing caches. During repeat training/evaluation, use `--no-auto_generate_normals` after confirming the cache is complete.

## Train FCFE-Net on University-1652

Example single-GPU training command:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ccr python -u train_university.py \
  --gpu_ids 0 \
  --name FCFE_Model_University \
  --save_dir ./model_fcfe \
  --data_dir /root/dataset/University-Release/train \
  --batchsize 8 \
  --h 384 \
  --w 384 \
  --block 2 \
  --M 32 \
  --sample_num 1 \
  --triplet_loss 0.3 \
  --lr 0.0001 \
  --epochs 80 \
  --steps 40 60 \
  --save_interval 5 \
  --save_after 5 \
  --model convnext_small_22k_224 \
  --use_fcfe \
  --use_normals \
  --normal_dir /root/dataset/University-Release/omnidata_normals_train \
  --no-auto_generate_normals \
  --num_workers 8
```

To fine-tune from an existing checkpoint, add:

```bash
--init_from_checkpoint ./model_fcfe/FCFE_Model_University/net_079.pth
```

Use separate single-GPU runs for parallel experiments instead of one multi-GPU run.

## Test FCFE-Net on University-1652

### Drone-to-Satellite (D2S)

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ccr python -u test_university.py \
  --gpu_ids 0 \
  --name FCFE_Model_University \
  --save_dir ./model_fcfe \
  --test_dir /root/dataset/University-Release/test \
  --which_epoch last \
  --mode 2 \
  --batchsize 64 \
  --num_workers 8 \
  --normal_dir /root/dataset/University-Release/omnidata_normals_test \
  --no-auto_generate_normals
```

### Satellite-to-Drone (S2D)

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ccr python -u test_university.py \
  --gpu_ids 0 \
  --name FCFE_Model_University \
  --save_dir ./model_fcfe \
  --test_dir /root/dataset/University-Release/test \
  --which_epoch last \
  --mode 1 \
  --batchsize 64 \
  --num_workers 8 \
  --normal_dir /root/dataset/University-Release/omnidata_normals_test \
  --no-auto_generate_normals
```

Evaluation writes the following files under the selected experiment directory:

```text
pytorch_result.mat
result.txt
evaluate.txt
```

Use `--which_epoch 029`, `--which_epoch 079`, etc. to evaluate a specific checkpoint.

## Train and test SUES-200

SUES-200 is trained directly in this project. Example command:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ccr python -u train_sues.py \
  --gpu_ids 0 \
  --name FCFE_Model_SUES \
  --data_dir /root/dataset/SUES-200-512x512/Training/200 \
  --batchsize 8 \
  --h 512 \
  --w 512 \
  --block 2 \
  --M 32 \
  --sample_num 1 \
  --triplet_loss 0.3 \
  --lr 0.01 \
  --epochs 80 \
  --steps 40 60
```

Test with:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n ccr python -u test_sues.py \
  --gpu_ids 0 \
  --name FCFE_Model_SUES \
  --test_dir /root/dataset/SUES-200-512x512/Testing/200 \
  --which_epoch last \
  --mode 2
```

## Important options

- `--use_fcfe`: enables the FCFE enhancement path.
- `--use_normals`: loads paired RGB/normal inputs.
- `--normal_dir`: path to the Omnidata normal cache.
- `--auto_generate_normals`: generate missing normal images on demand.
- `--no-auto_generate_normals`: require a complete normal cache and fail if a normal is missing.
- `--fcfe_dual_backbone`: use a separate normal-image backbone. Leave disabled unless memory allows.
- `--init_from_checkpoint`: initialize matching tensors from an existing checkpoint.
- `--save_dir`: root directory for experiment outputs.
- `--which_epoch`: checkpoint epoch to evaluate, or `last`.

## Verification

Run syntax and smoke checks in the `ccr` environment:

```bash
conda run -n ccr python -m py_compile \
  models/ConvNext/fcfe_modules.py \
  models/ConvNext/make_model.py \
  models/model.py \
  datasets/omnidata_normals.py \
  datasets/Dataloader_University.py \
  datasets/make_dataloader_university.py \
  datasets/normal_imagefolder.py \
  train_university.py \
  test_university.py \
  utils.py
```

```bash
conda run -n ccr python test_fcfe_smoke.py
```

## Notes on checkpoint compatibility

Older experiment directories and checkpoints can still be evaluated by explicitly passing their original `--name` and `--save_dir`. The current default names use `FCFE_*`, while historical output directories are left unchanged to preserve reproducibility.
