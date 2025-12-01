"""
简化的TCIA数据集加载器
专门用于PET图像二分类任务
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import glob

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    print("警告: nibabel未安装")
    NIBABEL_AVAILABLE = False


class TCIADataset(Dataset):
    """简化的TCIA数据集类"""
    
    def __init__(self, 
                 data_dir: str,
                 target_size: Tuple[int, int, int] = (64, 128, 128),
                 normalize: bool = True):
        """
        Args:
            data_dir: 数据目录路径，应包含normal/和cancer/子目录
            target_size: 目标尺寸 (depth, height, width)
            normalize: 是否标准化
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.normalize = normalize
        
        # 获取数据文件列表和标签
        self.data_files, self.labels = self._get_data_files()
        
    def _get_data_files(self) -> Tuple[List[str], List[int]]:
        """获取数据文件列表和对应标签"""
        data_files = []
        labels = []
        
        # 正常样本 (标签=0)
        normal_dir = os.path.join(self.data_dir, 'normal')
        if os.path.exists(normal_dir):
            normal_files = glob.glob(os.path.join(normal_dir, '*.nii.gz'))
            data_files.extend(normal_files)
            labels.extend([0] * len(normal_files))
            print(f"找到正常样本: {len(normal_files)}")
        
        # 癌症样本 (标签=1)
        cancer_dir = os.path.join(self.data_dir, 'cancer')
        if os.path.exists(cancer_dir):
            cancer_files = glob.glob(os.path.join(cancer_dir, '*.nii.gz'))
            data_files.extend(cancer_files)
            labels.extend([1] * len(cancer_files))
            print(f"找到癌症样本: {len(cancer_files)}")
        
        if len(data_files) == 0:
            raise ValueError(f"未在{self.data_dir}中找到数据文件")
            
        return data_files, labels
    
    def _load_volume(self, file_path: str) -> np.ndarray:
        """加载3D体积数据"""
        if not NIBABEL_AVAILABLE:
            raise ImportError("需要安装nibabel来加载NIfTI文件")
        
        # 加载NIfTI文件
        nii = nib.load(file_path)
        volume = nii.get_fdata()
        
        # 调整到目标尺寸
        volume = self._resize_volume(volume)
        
        # 标准化
        if self.normalize:
            volume = self._normalize_volume(volume)
        
        return volume
    
    def _resize_volume(self, volume: np.ndarray) -> np.ndarray:
        """调整体积大小到目标尺寸"""
        from scipy.ndimage import zoom
        
        current_shape = volume.shape
        target_shape = self.target_size
        
        # 计算缩放因子
        zoom_factors = [t/c for c, t in zip(current_shape, target_shape)]
        
        # 使用三线性插值调整大小
        resized = zoom(volume, zoom_factors, order=1)
        
        return resized
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """标准化体积数据"""
        # 去除异常值
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        
        # Z-score标准化
        mean = np.mean(volume)
        std = np.std(volume)
        if std > 0:
            volume = (volume - mean) / std
        
        return volume
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        label = self.labels[idx]
        
        # 加载体积数据
        volume = self._load_volume(file_path)
        
        # 转换为PyTorch张量
        volume = torch.from_numpy(volume).float().unsqueeze(0)  # 添加通道维度
        label = torch.tensor(label, dtype=torch.long)
        
        return volume, label


def create_data_loaders(data_dir: str, 
                       batch_size: int = 4,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       target_size: Tuple[int, int, int] = (64, 128, 128),
                       num_workers: int = 4,
                       seed: int = 42):
    """创建训练、验证和测试数据加载器"""
    
    # 创建完整数据集
    full_dataset = TCIADataset(
        data_dir=data_dir,
        target_size=target_size,
        normalize=True
    )
    
    # 获取所有文件路径和标签
    files = full_dataset.data_files
    labels = full_dataset.labels
    
    # 分层划分数据集
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        files, labels, train_size=train_ratio, stratify=labels, random_state=seed
    )
    
    # 计算验证和测试集的相对比例
    val_size = val_ratio / (val_ratio + test_ratio)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, train_size=val_size, stratify=temp_labels, random_state=seed
    )
    
    # 创建子数据集
    class SubDataset(Dataset):
        def __init__(self, files, labels, base_dataset):
            self.files = files
            self.labels = labels
            self.base_dataset = base_dataset
        
        def __len__(self):
            return len(self.files)
        
        def __getitem__(self, idx):
            file_path = self.files[idx]
            label = self.labels[idx]
            
            # 重用基础数据集的加载逻辑
            volume = self.base_dataset._load_volume(file_path)
            volume = torch.from_numpy(volume).float().unsqueeze(0)
            label = torch.tensor(label, dtype=torch.long)
            
            return volume, label
    
    train_dataset = SubDataset(train_files, train_labels, full_dataset)
    val_dataset = SubDataset(val_files, val_labels, full_dataset)
    test_dataset = SubDataset(test_files, test_labels, full_dataset)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"数据集划分完成:")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    return train_loader, val_loader, test_loader
