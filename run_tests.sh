#!/bin/bash

# Qwen3-30B-A3B 模型测试运行脚本

echo "=========================================="
echo "Qwen3-30B-A3B 模型测试套件"
echo "=========================================="

# 检查Python环境
echo "检查Python环境..."
python3 --version
if [ $? -ne 0 ]; then
    echo "错误: Python3 未安装"
    exit 1
fi

# 检查必要的包
echo "检查必要的Python包..."
python3 -c "import torch, vllm, GPUtil, psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: 缺少必要的Python包"
    echo "请运行: pip install torch vllm GPUtil psutil"
    exit 1
fi

# 检查GPU
echo "检查GPU状态..."
nvidia-smi
if [ $? -ne 0 ]; then
    echo "警告: nvidia-smi 命令失败，可能没有NVIDIA GPU"
fi

# 检查模型文件
MODEL_PATH="./models/models--Qwen--Qwen3-30B-A3B/snapshots/ae659febe817e4b3ebd7355f47792725801204c9"
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    echo "请确保已下载模型到指定路径"
    exit 1
fi

echo "模型路径检查通过: $MODEL_PATH"

# 选择测试类型
echo ""
echo "请选择测试类型:"
echo "1. 快速测试 (验证模型是否正常工作)"
echo "2. 完整性能测试 (包含所有测试用例和性能评估)"
echo "3. 两者都运行"
read -p "请输入选择 (1/2/3): " choice

case $choice in
    1)
        echo "运行快速测试..."
        python3 quick_model_test.py
        ;;
    2)
        echo "运行完整性能测试..."
        python3 offline_model_test.py
        ;;
    3)
        echo "首先运行快速测试..."
        python3 quick_model_test.py
        if [ $? -eq 0 ]; then
            echo ""
            echo "快速测试通过，现在运行完整测试..."
            python3 offline_model_test.py
        else
            echo "快速测试失败，跳过完整测试"
        fi
        ;;
    *)
        echo "无效选择，退出"
        exit 1
        ;;
esac

echo ""
echo "测试完成!"
