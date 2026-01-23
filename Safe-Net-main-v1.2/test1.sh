#!/bin/bash

# 测试配置参数
name="SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652"  # 与训练时的name保持一致（用于定位checkpoint目录）
test_dir="/datasets/University-Release/test"  # 测试数据集路径
gpu_ids=0                                       # 使用的GPU编号
num_worker=4                                    # 数据加载线程数
batchsize=128                                   # 测试批次大小
pad=0                                           # 图像填充参数
h=256                                           # 图像高度
w=256                                           # 图像宽度
start_epoch=119                                 # 起始测试的epoch
end_epoch=120                                   # 结束测试的epoch
step_epoch=20                                   # 测试epoch间隔

# 进入checkpoint目录（模型权重保存路径）
cd checkpoints

# 循环测试不同epoch的模型权重
for ((i=start_epoch; i<=end_epoch; i+=step_epoch)); do
  # 循环测试不同填充参数（原脚本逻辑保留，可根据需要调整）
  for ((p=0; p<=pad; p+=10)); do
    # 循环测试两种模式（1: drone->satellite  2: satellite->drone）
    for ((mode=1; mode<3; mode++)); do
      echo "----------------------------------------"
      echo "Testing epoch: $i, pad: $p, mode: $mode"
      python ../test.py \
        --test_dir "$test_dir" \
        --checkpoint "net_$i.pth" \
        --mode $mode \
        --gpu_ids $gpu_ids \
        --num_worker $num_worker \
        --batchsize $batchsize \
        --pad $p \
        --h $h \
        --w $w
      echo "----------------------------------------"
    done
  done
done

echo "All tests completed!"