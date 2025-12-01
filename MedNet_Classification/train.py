"""
MedNet训练器 - 用于TCIA数据集上的癌症二分类
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from models.mednet import get_pretrained_model, mednet34, MEDICAL_NET_MODELS
from data.dataset import create_data_loaders
from mednet_config import get_config, get_pretrained_path


class MedNetTrainer:
    """
    MedNet训练器类 - 支持MedicalNet预训练模型
    """
    
    def __init__(self, 
                 data_dir: str,
                 csv_file: str = None,
                 model_name: str = 'resnet_34',
                 num_classes: int = 2,
                 batch_size: int = 4,
                 learning_rate: float = 0.001,
                 num_epochs: int = 100,
                 device: str = 'auto',
                 output_dir: str = './outputs',
                 early_stopping_patience: int = 15,
                 target_size: Tuple[int, int, int] = (64, 128, 128),
                 class_weights: Optional[torch.Tensor] = None,
                 use_pretrained: bool = True,
                 pretrained_path: str = None):
        """
        初始化训练器
        
        Args:
            data_dir: 数据目录路径
            csv_file: 标签CSV文件路径
            model_name: MedicalNet模型名称 ('resnet_10', 'resnet_18', 'resnet_34', 'resnet_50', 等)
            num_classes: 类别数量
            batch_size: 批次大小
            learning_rate: 学习率
            num_epochs: 训练轮数
            device: 设备 ('auto', 'cpu', 'cuda')
            output_dir: 输出目录
            early_stopping_patience: 早停耐心值
            target_size: 目标图像尺寸
            class_weights: 类别权重
            use_pretrained: 是否使用预训练权重
            pretrained_path: 预训练模型路径
        """
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.model_name = model_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.target_size = target_size
        self.class_weights = class_weights
        self.use_pretrained = use_pretrained
        self.pretrained_path = pretrained_path
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建模型
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            data_dir=data_dir,
            csv_file=csv_file,
            batch_size=batch_size,
            target_size=target_size
        )
        
        # 获取类别权重
        if class_weights is None:
            self.class_weights = self.train_loader.dataset.dataset.get_class_weights()
        self.class_weights = self.class_weights.to(self.device)
        
        # 创建损失函数和优化器
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=7, verbose=True
        )
        
        # 初始化记录器
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}
        
        # 早停相关
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
    def _create_model(self):
        """创建模型"""
        try:
            if self.use_pretrained:
                # 尝试获取预训练路径
                if self.pretrained_path is None:
                    try:
                        self.pretrained_path = get_pretrained_path(self.model_name)
                    except:
                        print(f"警告: 无法获取预训练模型路径，使用随机初始化")
                        self.pretrained_path = None
                
                # 使用预训练模型
                model = get_pretrained_model(
                    model_name=self.model_name,
                    num_classes=self.num_classes,
                    input_channels=1,
                    pretrained_dir='pretrained_models'
                )
                print(f"使用预训练模型: {self.model_name}")
            else:
                # 使用随机初始化的模型
                if self.model_name in ['resnet_34', 'mednet34']:
                    model = mednet34(num_classes=self.num_classes, input_channels=1)
                elif self.model_name in ['resnet_18', 'mednet18']:  
                    from models.mednet import mednet18
                    model = mednet18(num_classes=self.num_classes, input_channels=1)
                elif self.model_name in ['resnet_50', 'mednet50']:
                    from models.mednet import mednet50
                    model = mednet50(num_classes=self.num_classes, input_channels=1)
                else:
                    # 默认使用ResNet-34
                    model = mednet34(num_classes=self.num_classes, input_channels=1)
                    print(f"警告: 未知模型名称 {self.model_name}，使用默认的ResNet-34")
                
                print(f"使用随机初始化模型: {self.model_name}")
            
            return model
            
        except Exception as e:
            print(f"模型创建失败: {e}")
            print("使用默认的ResNet-34模型")
            return mednet34(num_classes=self.num_classes, input_channels=1)
    
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f'Batch [{batch_idx + 1}/{len(self.train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> Tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self):
        """主训练循环"""
        print("开始训练...")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            print(f'\nEpoch [{epoch + 1}/{self.num_epochs}]')
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate_epoch()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.val_history['loss'].append(val_loss)
            self.val_history['accuracy'].append(val_acc)
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印结果
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                self.early_stopping_counter = 0
                print(f'保存最佳模型 (Val Loss: {val_loss:.4f})')
            else:
                self.early_stopping_counter += 1
            
            # 早停检查
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f'Early stopping after {epoch + 1} epochs')
                break
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        training_time = time.time() - start_time
        print(f'\n训练完成，总时间: {training_time / 60:.2f} 分钟')
        
        # 保存训练历史
        self.save_training_history()
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 在测试集上评估
        self.evaluate_on_test_set()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': {
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'target_size': self.target_size
            }
        }
        
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
        else:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch + 1}.pth')
    
    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_loss': self.train_history['loss'],
            'train_accuracy': self.train_history['accuracy'],
            'val_loss': self.val_history['loss'],
            'val_accuracy': self.val_history['accuracy']
        }
        
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 损失曲线
        epochs = range(1, len(self.train_history['loss']) + 1)
        ax1.plot(epochs, self.train_history['loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_history['loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(epochs, self.train_history['accuracy'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_history['accuracy'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"训练曲线已保存到: {self.output_dir / 'training_curves.png'}")
    
    def evaluate_on_test_set(self):
        """在测试集上评估模型"""
        print("\n在测试集上评估模型...")
        
        # 加载最佳模型
        checkpoint = torch.load(self.output_dir / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 计算指标
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # AUC (如果是二分类)
        if self.num_classes == 2:
            auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        
        # 打印结果
        print(f"测试集结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        if self.num_classes == 2:
            print(f"AUC: {auc:.4f}")
        
        # 保存结果
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        if self.num_classes == 2:
            results['auc'] = float(auc)
        
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(cm)
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        
        class_names = ['Normal', 'Cancer'] if self.num_classes == 2 else [f'Class {i}' for i in range(self.num_classes)]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"混淆矩阵已保存到: {self.output_dir / 'confusion_matrix.png'}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint.get('train_history', {'loss': [], 'accuracy': []})
        self.val_history = checkpoint.get('val_history', {'loss': [], 'accuracy': []})
        
        print(f"已加载检查点: {checkpoint_path}")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        
        return checkpoint['epoch']


if __name__ == "__main__":
    # 配置参数
    config = {
        'data_dir': '/path/to/tcia/data',  # 替换为实际数据路径
        'csv_file': None,  # 如果有标签文件，请提供路径
        'model_name': 'mednet18',  # 'mednet18', 'mednet34', 'mednet50'
        'batch_size': 4,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'output_dir': './outputs',
        'target_size': (64, 128, 128)  # (depth, height, width)
    }
    
    # 创建训练器
    trainer = MedNetTrainer(**config)
    
    # 开始训练
    trainer.train()
