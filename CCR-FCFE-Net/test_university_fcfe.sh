#!/bin/bash
set -e

# ============================================
# FCFE-Net University1652 inference script
# ============================================

# GPU id visible to this script. Keep one GPU per run.
gpu_id="2"

# Experiment settings. Change name/epoch to evaluate another checkpoint.
save_dir="./model_fcfe"
name="CCR_FCFE_University_Gated_Continue_lr0002_from_lr001_079"
which_epoch="039"

# Dataset and Omnidata normal cache.
test_dir="/root/dataset/University-Release/test"
normal_dir="/root/dataset/University-Release/omnidata_normals_test"

# Inference settings.
batchsize=64
num_workers=8

if [ ! -d "$test_dir" ]; then
  echo "Error: test_dir does not exist: $test_dir"
  exit 1
fi

if [ ! -d "$normal_dir" ]; then
  echo "Error: normal_dir does not exist: $normal_dir"
  echo "Please generate Omnidata normals first, or change normal_dir."
  exit 1
fi

if [ ! -d "$save_dir/$name" ]; then
  echo "Error: model directory does not exist: $save_dir/$name"
  echo "Set name/save_dir to an existing experiment before running."
  exit 1
fi

echo "============================================"
echo "FCFE-Net University1652 inference"
echo "Model: $save_dir/$name"
echo "Epoch: $which_epoch"
echo "Test data: $test_dir"
echo "Normal cache: $normal_dir"
echo "GPU: $gpu_id"
echo "============================================"

echo ""
echo "========== D2S: Drone -> Satellite =========="
CUDA_VISIBLE_DEVICES="$gpu_id" conda run --no-capture-output -n ccr python -u test_university.py \
  --gpu_ids 0 \
  --name "$name" \
  --save_dir "$save_dir" \
  --test_dir "$test_dir" \
  --which_epoch "$which_epoch" \
  --mode 2 \
  --batchsize "$batchsize" \
  --num_workers "$num_workers" \
  --normal_dir "$normal_dir" \
  --no-auto_generate_normals

echo ""
echo "========== S2D: Satellite -> Drone =========="
CUDA_VISIBLE_DEVICES="$gpu_id" conda run --no-capture-output -n ccr python -u test_university.py \
  --gpu_ids 0 \
  --name "$name" \
  --save_dir "$save_dir" \
  --test_dir "$test_dir" \
  --which_epoch "$which_epoch" \
  --mode 1 \
  --batchsize "$batchsize" \
  --num_workers "$num_workers" \
  --normal_dir "$normal_dir" \
  --no-auto_generate_normals

echo ""
echo "Inference finished."
echo "Results: $save_dir/$name/result.txt and $save_dir/$name/evaluate.txt"
