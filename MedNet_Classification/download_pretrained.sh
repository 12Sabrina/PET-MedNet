#!/bin/bash

# MedicalNet ResNet-34 预训练模型下载脚本
# 专门用于TCIA PET数据的二分类任务

# 创建预训练模型目录
mkdir -p pretrained_models
cd pretrained_models

echo "=========================================="
echo "下载MedicalNet ResNet-34预训练模型"
echo "用于TCIA PET数据二分类任务"
echo "=========================================="

# 检查是否安装了gdown
if ! command -v gdown &> /dev/null; then
    echo "正在安装gdown..."
    pip install gdown
fi

echo "下载ResNet-34预训练模型..."
# 注意：请替换为真实的Google Drive ID或直接下载链接
gdown --id "1KiJ8lVvVLlakZd9gEiI5gNQdG77Rw1CP" -O resnet_34_23dataset.pth

if [ $? -eq 0 ]; then
    echo "✅ ResNet-34下载完成!"
    echo "模型文件: $(pwd)/resnet_34_23dataset.pth"
else
    echo "❌ 下载失败，请检查网络连接或Google Drive链接"
    exit 1
fi
