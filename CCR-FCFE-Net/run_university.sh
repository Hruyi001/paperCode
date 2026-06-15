name="FCFE_Model_University"
# ============================================
# 数据路径配置 - 请根据您的实际数据集路径修改
# ============================================
# 训练数据路径（应包含 satellite/, drone/, street/ 等子目录）
data_dir="/root/dataset/University-Release/train"
# 测试数据路径（应包含 query_satellite/, query_drone/, gallery_satellite/, gallery_drone/ 等子目录）
test_dir="/root/dataset/University-Release/test"

# 如果上面的路径不存在，请尝试以下常见路径：
# data_dir="/root/dataset/University-Release/train"
# test_dir="/root/dataset/University-Release/test"
# 或者
# data_dir="../data/University-Release/train"
# test_dir="../data/University-Release/test"

# ============================================
# 训练参数配置
# ============================================
gpu_ids="0"
lr=0.01
batchsize=8
triplet_loss=0.3
num_epochs=200
views=2
M=32
# 指定模型保存路径，避免覆盖现有模型
save_dir="./model_new"  # 可以修改为您想要的路径，例如: "./model_$(date +%Y%m%d_%H%M%S)"

# ============================================
# 路径检查
# ============================================
echo "检查数据路径..."
if [ ! -d "$data_dir" ]; then
    echo "错误: 训练数据路径不存在: $data_dir"
    echo "请修改脚本中的 data_dir 变量为正确的路径"
    exit 1
fi

if [ ! -d "$data_dir/satellite" ]; then
    echo "错误: 训练数据目录中缺少 satellite/ 子目录"
    echo "请确认数据集结构是否正确"
    exit 1
fi

if [ ! -d "$test_dir" ]; then
    echo "警告: 测试数据路径不存在: $test_dir"
    echo "测试阶段可能会失败，但训练可以继续"
fi

echo "数据路径检查通过，开始训练..."
echo "训练数据路径: $data_dir"
echo "测试数据路径: $test_dir"
echo "模型保存路径: $save_dir"


python train_university.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --views $views --lr $lr \
 --batchsize $batchsize --triplet_loss $triplet_loss --epochs $num_epochs --M $M --save_dir $save_dir \

for ((j = 1; j < 3; j++));
    do
      python test_university.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --mode $j
    done
