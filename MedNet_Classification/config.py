"""
简化配置文件 - TCIA PET数据二分类任务
"""

import os

# 项目配置
PROJECT_NAME = "TCIA_PET_Classification"

# 数据配置
DATA_CONFIG = {
    'data_dir': '/path/to/tcia/pet/data',  # 请替换为实际的TCIA数据路径
    'target_size': (64, 128, 128),  # (depth, height, width)
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'seed': 42
}

# 模型配置
MODEL_CONFIG = {
    'model_name': 'resnet_34',  # 使用ResNet-34
    'num_classes': 2,  # 二分类
    'pretrained_path': './pretrained_models/resnet_34_23dataset.pth'
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 4,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'early_stopping_patience': 10,
    'num_workers': 4
}

# 输出配置
OUTPUT_CONFIG = {
    'output_dir': './outputs',
    'save_model_path': './outputs/best_model.pth'
}

def get_config():
    """获取配置"""
    return {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'output': OUTPUT_CONFIG
    }
