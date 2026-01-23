#!/bin/bash

# 训练配置参数
name="SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652"
data_dir="/datasets/University-Release/train"
gpu_ids=0
num_worker=4
lr=0.01
sample_num=2
block=4
batchsize=8
triplet_loss=0
num_epochs=120
views=2
h=256
w=256

# 运行训练
# 模型会在第30、60、90、120轮自动保存
python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --num_worker $num_worker --views $views --lr $lr \
--sample_num $sample_num --block $block --batchsize $batchsize --triplet_loss $triplet_loss --num_epochs $num_epochs --h $h --w $w

echo "训练完成！模型保存在 checkpoints/$name/ 目录下"
