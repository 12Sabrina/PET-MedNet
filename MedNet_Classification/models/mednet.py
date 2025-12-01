"""
MedNet模型实现 - 基于MedicalNet的3D CNN用于医学图像分析
原论文: "Med3D: Transfer Learning for 3D Medical Image Analysis"
适配ACRIN-FMISO-BRAIN数据集的二分类任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional, Dict
import warnings
from functools import partial


class BasicBlock3D(nn.Module):
    """3D基础残差块"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    """3D瓶颈残差块"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def downsample_basic_block(x, planes, stride, no_cuda=False):
    """基础块下采样函数"""
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = torch.cat([out.data, zero_pads], dim=1)
    
    return out


class MedNet(nn.Module):
    """
    MedNet 3D CNN模型 - 基于MedicalNet项目
    适配ACRIN-FMISO-BRAIN数据集的二分类任务
    """
    
    def __init__(self, block, layers, num_classes=2, input_channels=1, dropout=0.5,
                 shortcut_type='B', widen_factor=1.0, no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(MedNet, self).__init__()
        
        # 输入层 - 与MedicalNet保持一致
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=(2, 2, 2), 
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride,
                                     no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
    
    def load_medical_pretrained(self, pretrained_path: str, strict: bool = False):
        """
        加载MedicalNet预训练权重
        
        Args:
            pretrained_path: 预训练模型路径
            strict: 是否严格匹配所有参数
        """
        if not os.path.exists(pretrained_path):
            warnings.warn(f"预训练模型文件不存在: {pretrained_path}")
            return
        
        try:
            # 加载预训练权重
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # 处理不同的checkpoint格式
            if 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                pretrained_dict = checkpoint['model']
            else:
                pretrained_dict = checkpoint
            
            # 过滤掉分类层的权重（因为类别数可能不同）
            filtered_dict = {}
            model_dict = self.state_dict()
            
            for k, v in pretrained_dict.items():
                # 移除module.前缀（如果存在）
                if k.startswith('module.'):
                    k = k[7:]
                
                # 跳过最后的分类层
                if 'fc.' in k:
                    continue
                    
                if k in model_dict and v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    print(f"跳过参数: {k} (形状不匹配或不存在)")
            
            # 加载过滤后的权重
            self.load_state_dict(filtered_dict, strict=False)
            print(f"成功加载预训练权重: {len(filtered_dict)} 个参数")
            
        except Exception as e:
            warnings.warn(f"加载预训练模型失败: {e}")


# 模型构建函数
def mednet10(num_classes=2, input_channels=1, pretrained_path=None, **kwargs):
    """构建MedNet-10模型"""
    model = MedNet(BasicBlock3D, [1, 1, 1, 1], num_classes=num_classes, 
                   input_channels=input_channels, **kwargs)
    if pretrained_path:
        model.load_medical_pretrained(pretrained_path)
    return model

def mednet18(num_classes=2, input_channels=1, pretrained_path=None, **kwargs):
    """构建MedNet-18模型"""
    model = MedNet(BasicBlock3D, [2, 2, 2, 2], num_classes=num_classes, 
                   input_channels=input_channels, **kwargs)
    if pretrained_path:
        model.load_medical_pretrained(pretrained_path)
    return model

def mednet34(num_classes=2, input_channels=1, pretrained_path=None, **kwargs):
    """构建MedNet-34模型 (推荐用于FMISO数据)"""
    model = MedNet(BasicBlock3D, [3, 4, 6, 3], num_classes=num_classes, 
                   input_channels=input_channels, **kwargs)
    if pretrained_path:
        model.load_medical_pretrained(pretrained_path)
    return model

def mednet50(num_classes=2, input_channels=1, pretrained_path=None, **kwargs):
    """构建MedNet-50模型"""
    model = MedNet(Bottleneck3D, [3, 4, 6, 3], num_classes=num_classes, 
                   input_channels=input_channels, **kwargs)
    if pretrained_path:
        model.load_medical_pretrained(pretrained_path)
    return model

def mednet101(num_classes=2, input_channels=1, pretrained_path=None, **kwargs):
    """构建MedNet-101模型"""
    model = MedNet(Bottleneck3D, [3, 4, 23, 3], num_classes=num_classes, 
                   input_channels=input_channels, **kwargs)
    if pretrained_path:
        model.load_medical_pretrained(pretrained_path)
    return model

def mednet152(num_classes=2, input_channels=1, pretrained_path=None, **kwargs):
    """构建MedNet-152模型"""
    model = MedNet(Bottleneck3D, [3, 8, 36, 3], num_classes=num_classes, 
                   input_channels=input_channels, **kwargs)
    if pretrained_path:
        model.load_medical_pretrained(pretrained_path)
    return model

def mednet200(num_classes=2, input_channels=1, pretrained_path=None, **kwargs):
    """构建MedNet-200模型"""
    model = MedNet(Bottleneck3D, [3, 24, 36, 3], num_classes=num_classes, 
                   input_channels=input_channels, **kwargs)
    if pretrained_path:
        model.load_medical_pretrained(pretrained_path)
    return model


# MedicalNet预训练模型下载链接和配置
MEDICAL_NET_MODELS = {
    'resnet_10': {
        'url': 'https://drive.google.com/uc?id=1--s6wN2vN9-8-5J7S0X3tHqjz9e-l2',
        'filename': 'resnet_10_23dataset.pth',
        'architecture': 'mednet10',
        'description': '10层ResNet，参数量最少(14.36M)，适合快速实验'
    },
    'resnet_18': {
        'url': 'https://drive.google.com/uc?id=1-WGsSDNqN5pPy1yksEcKOTl5qE0I6d8',  
        'filename': 'resnet_18_23dataset.pth',
        'architecture': 'mednet18',
        'description': '18层ResNet(32.99M)，良好的性能平衡'
    },
    'resnet_34': {
        'url': 'https://drive.google.com/uc?id=15ouNNnPm2jPtGU-rYXsYWS8gP6yEUOhF',
        'filename': 'resnet_34_23dataset.pth', 
        'architecture': 'mednet34',
        'description': '34层ResNet(63.31M)，推荐用于FMISO数据集'
    },
    'resnet_50': {
        'url': 'https://drive.google.com/uc?id=1jDzyWDtD9R3mZd5v8h3ZYVWqKNg7o-5',
        'filename': 'resnet_50_23dataset.pth',
        'architecture': 'mednet50', 
        'description': '50层ResNet(46.21M)，更深的网络'
    },
    'resnet_101': {
        'url': 'https://drive.google.com/uc?id=1PgMi1zJXOAgX4H9HyEm_1YFw_CHw1m7',
        'filename': 'resnet_101_23dataset.pth',
        'architecture': 'mednet101',
        'description': '101层ResNet(85.31M)，参数较多'
    },
    'resnet_152': {
        'url': 'https://drive.google.com/uc?id=1zq4Jw_nQ5ZXyJhfcqZQ2VLvD7DH4_Q8',
        'filename': 'resnet_152_23dataset.pth',
        'architecture': 'mednet152',
        'description': '152层ResNet(117.51M)，最深的网络'
    },
    'resnet_200': {
        'url': 'https://drive.google.com/uc?id=1yaq7oi2XXDjPB8YLVNv8VbL5hd9c_z8',
        'filename': 'resnet_200_23dataset.pth',
        'architecture': 'mednet200',
        'description': '200层ResNet(126.74M)，极深网络'
    }
}


def get_pretrained_model(model_name: str = 'resnet_34', 
                        num_classes: int = 2,
                        input_channels: int = 1,
                        pretrained_dir: str = 'pretrained_models',
                        download: bool = False,
                        **kwargs) -> MedNet:
    """
    获取预训练的MedNet模型
    
    Args:
        model_name: 模型名称 ('resnet_10', 'resnet_18', 'resnet_34', 等)
        num_classes: 分类数量
        input_channels: 输入通道数  
        pretrained_dir: 预训练模型存储目录
        download: 是否自动下载模型（需要网络连接）
        **kwargs: 其他模型参数
        
    Returns:
        预训练的MedNet模型
    """
    if model_name not in MEDICAL_NET_MODELS:
        raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {list(MEDICAL_NET_MODELS.keys())}")
    
    model_info = MEDICAL_NET_MODELS[model_name]
    architecture = model_info['architecture']
    filename = model_info['filename']
    
    # 创建预训练模型目录
    os.makedirs(pretrained_dir, exist_ok=True)
    pretrained_path = os.path.join(pretrained_dir, filename)
    
    # 检查是否已下载
    if not os.path.exists(pretrained_path) and download:
        print(f"下载预训练模型: {model_name}")
        print(f"描述: {model_info['description']}")
        download_pretrained_model(model_info['url'], pretrained_path)
    
    # 构建模型
    if architecture == 'mednet10':
        model = mednet10(num_classes=num_classes, input_channels=input_channels, **kwargs)
    elif architecture == 'mednet18':
        model = mednet18(num_classes=num_classes, input_channels=input_channels, **kwargs)
    elif architecture == 'mednet34':
        model = mednet34(num_classes=num_classes, input_channels=input_channels, **kwargs)
    elif architecture == 'mednet50':
        model = mednet50(num_classes=num_classes, input_channels=input_channels, **kwargs)
    elif architecture == 'mednet101':
        model = mednet101(num_classes=num_classes, input_channels=input_channels, **kwargs)
    elif architecture == 'mednet152':
        model = mednet152(num_classes=num_classes, input_channels=input_channels, **kwargs)
    elif architecture == 'mednet200':
        model = mednet200(num_classes=num_classes, input_channels=input_channels, **kwargs)
    else:
        raise ValueError(f"未知的架构: {architecture}")
    
    # 加载预训练权重
    if os.path.exists(pretrained_path):
        model.load_medical_pretrained(pretrained_path)
        print(f"已加载预训练模型: {pretrained_path}")
    else:
        print(f"预训练模型文件不存在: {pretrained_path}")
        print("将使用随机初始化的权重")
        
    return model


def download_pretrained_model(url: str, save_path: str):
    """
    下载预训练模型
    注意：这是一个占位函数，实际使用时需要实现具体的下载逻辑
    """
    print(f"请手动下载预训练模型:")
    print(f"URL: {url}")
    print(f"保存路径: {save_path}")
    print("或者使用以下命令下载:")
    print(f"wget -O {save_path} '{url}'")
    print(f"或使用gdown: gdown --id <file_id> -O {save_path}")


def print_model_info():
    """打印所有可用模型的信息"""
    print("=" * 80)
    print("MedicalNet预训练模型信息")
    print("=" * 80)
    
    for name, info in MEDICAL_NET_MODELS.items():
        print(f"\n模型名称: {name}")
        print(f"文件名: {info['filename']}")
        print(f"架构: {info['architecture']}")
        print(f"描述: {info['description']}")
        print(f"下载链接: {info['url']}")
    
    print("\n推荐用于ACRIN-FMISO-BRAIN数据集: resnet_34")
    print("平衡了模型复杂度和性能，参数量适中 (63.31M)")


if __name__ == "__main__":
    # 测试模型构建
    print("测试MedNet模型构建...")
    
    # 打印模型信息
    print_model_info()
    
    # 构建推荐的模型
    print(f"\n构建MedNet-34模型...")
    model = mednet34(num_classes=2, input_channels=1)
    
    # 测试前向传播
    print("测试前向传播...")
    x = torch.randn(1, 1, 64, 128, 128)  # FMISO PET数据尺寸
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    print("\nMedNet模型测试完成!")
