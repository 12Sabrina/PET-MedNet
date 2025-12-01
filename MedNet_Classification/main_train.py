#!/usr/bin/env python3
"""
TCIA PET数据集上的MedNet训练主脚本
使用MedicalNet预训练模型进行迁移学习
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import MedNetTrainer


def set_seed(seed: int = 42):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MedNet TCIA PET Classification Training')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, 
                       default='./tcia_data/organized_data',
                       help='TCIA数据集路径')
    parser.add_argument('--csv_file', type=str, default=None,
                       help='标签CSV文件路径')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='resnet_34',
                       choices=['resnet_10', 'resnet_18', 'resnet_34', 'resnet_50'],
                       help='模型名称')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='早停耐心值')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='训练设备')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录')
    
    # 其他参数
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打印配置信息
    print("\n=== TCIA PET二分类训练 ===")
    print(f"数据目录: {args.data_dir}")
    print(f"模型: {args.model_name}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"输出目录: {args.output_dir}")
    print(f"设备: {args.device}")
    print("=" * 40)
    
    # 创建训练器
    trainer = MedNetTrainer(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        output_dir=args.output_dir,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # 开始训练
    print("\n🚀 开始训练...")
    trainer.train()
    
    print("\n✅ 训练完成！")
    print(f"模型和日志保存在: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        raise
