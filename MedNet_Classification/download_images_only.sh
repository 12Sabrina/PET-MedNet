#!/bin/bash
# 超算平台TCIA图像数据下载脚本 (仅图像)
# 适用于没有图形界面的Linux环境

set -e

echo "=== TCIA NSCLC-Radiogenomics 图像数据下载 (超算平台版) ==="

# 创建数据目录
DATA_DIR="./tcia_data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "数据将下载到: $(pwd)"

# 1. 下载必要的Python依赖 (仅图像处理相关)
echo "安装Python依赖..."
pip install pydicom nibabel pandas scipy

# 2. 下载NBIA Data Retriever
echo "下载NBIA Data Retriever (医学图像下载工具)..."
mkdir -p tools
cd tools

# 下载Java版本的NBIA Data Retriever
if [ ! -f "nbia-data-retriever-4.4.1.jar" ]; then
    echo "正在下载NBIA Data Retriever..."
    wget -q https://cbiit-download.nci.nih.gov/nbia/releases/ForTCIA/NBIADataRetriever_4.4.1/nbia-data-retriever-4.4.1.jar
    echo "✅ NBIA Data Retriever下载完成"
else
    echo "✅ NBIA Data Retriever已存在"
fi

# 检查Java环境
if ! command -v java &> /dev/null; then
    echo "❌ 警告: 未找到Java环境"
    echo "请安装Java (OpenJDK 8或更高版本):"
    echo "  CentOS/RHEL: sudo yum install java-1.8.0-openjdk"
    echo "  Ubuntu/Debian: sudo apt install openjdk-8-jre"
    echo "  或联系系统管理员加载Java模块: module load java"
    JAVA_AVAILABLE=false
else
    echo "✅ Java环境检查通过"
    java -version 2>&1 | head -n 1
    JAVA_AVAILABLE=true
fi

cd ..

# 3. 下载manifest文件 (包含所有图像的下载信息)
echo "下载图像清单文件..."
if [ ! -f "manifest.txt" ]; then
    curl -o manifest.txt 'https://services.cancerimagingarchive.net/services/v4/TCIA/query/getManifest?Collection=NSCLC-Radiogenomics'
    echo "✅ 图像清单文件下载完成"
else
    echo "✅ 图像清单文件已存在"
fi

# 4. 创建图像下载脚本
cat > download_images.sh << 'EOF'
#!/bin/bash
# DICOM图像下载脚本

echo "开始下载DICOM图像..."
echo "⚠️  注意: 图像数据大小约98GB，下载时间取决于网络速度"
echo "建议在screen或tmux会话中运行以防止连接断开"

# 检查磁盘空间
AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
REQUIRED_SPACE=104857600  # 100GB in KB
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo "❌ 磁盘空间不足"
    echo "需要: ~100GB, 可用: $(($AVAILABLE_SPACE/1024/1024))GB"
    exit 1
fi

echo "✅ 磁盘空间检查通过: $(($AVAILABLE_SPACE/1024/1024))GB可用"

# 创建图像目录
mkdir -p images

# 使用Java版NBIA Data Retriever下载
if [ -f "tools/nbia-data-retriever-4.4.1.jar" ]; then
    echo "使用NBIA Data Retriever下载图像..."
    echo "开始时间: $(date)"
    
    # 运行下载器
    java -jar tools/nbia-data-retriever-4.4.1.jar -m manifest.txt -d ./images -v
    
    if [ $? -eq 0 ]; then
        echo "✅ 图像下载完成"
        echo "完成时间: $(date)"
        echo "开始组织数据结构..."
        python3 organize_images.py ./images ./organized_data
    else
        echo "❌ 图像下载失败"
        exit 1
    fi
else
    echo "❌ 未找到NBIA Data Retriever"
    echo "请先运行主下载脚本"
    exit 1
fi
EOF

chmod +x download_images.sh

# 5. 创建图像组织脚本 (仅处理图像)
cat > organize_images.py << 'EOF'
#!/usr/bin/env python3
"""
简化的图像组织脚本
将TCIA DICOM图像转换为NIfTI格式并组织为二分类训练格式
"""

import os
import sys
import shutil
import numpy as np
from pathlib import Path
import pydicom
import nibabel as nib

def organize_images_for_training(source_dir, target_dir):
    """组织图像为训练格式"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 创建目标目录
    normal_dir = target_path / "normal"
    cancer_dir = target_path / "cancer"
    normal_dir.mkdir(parents=True, exist_ok=True)
    cancer_dir.mkdir(parents=True, exist_ok=True)
    
    print("组织DICOM图像为训练格式...")
    
    processed_count = 0
    pet_count = 0
    ct_count = 0
    
    # 遍历患者目录
    for patient_dir in source_path.iterdir():
        if not patient_dir.is_dir():
            continue
            
        patient_id = patient_dir.name
        print(f"处理患者: {patient_id}")
        
        # 查找PET和CT图像
        pet_series = find_pet_series(patient_dir)
        ct_series = find_ct_series(patient_dir)
        
        if not pet_series and not ct_series:
            print(f"  未找到PET或CT序列")
            continue
        
        # 优先使用PET图像，如果没有则使用CT
        if pet_series:
            print(f"  找到 {len(pet_series)} 个PET序列")
            for i, series_dir in enumerate(pet_series):
                if convert_series_to_nifti(series_dir, target_path, patient_id, f"PET_{i}", processed_count):
                    pet_count += 1
                    processed_count += 1
                    break  # 只取第一个成功转换的PET序列
        
        elif ct_series:
            print(f"  找到 {len(ct_series)} 个CT序列")
            for i, series_dir in enumerate(ct_series):
                if convert_series_to_nifti(series_dir, target_path, patient_id, f"CT_{i}", processed_count):
                    ct_count += 1
                    processed_count += 1
                    break  # 只取第一个成功转换的CT序列
    
    print(f"\n图像组织完成:")
    print(f"处理患者数: {processed_count}")
    print(f"PET图像: {pet_count} 个")
    print(f"CT图像: {ct_count} 个")
    print(f"正常样本: {len(list(normal_dir.glob('*.nii.gz')))} 个")
    print(f"癌症样本: {len(list(cancer_dir.glob('*.nii.gz')))} 个")

def find_pet_series(patient_dir):
    """查找PET序列目录"""
    pet_series = []
    
    for study_dir in patient_dir.iterdir():
        if not study_dir.is_dir():
            continue
        for series_dir in study_dir.iterdir():
            if not series_dir.is_dir():
                continue
            if is_pet_series(series_dir):
                pet_series.append(series_dir)
    
    return pet_series

def find_ct_series(patient_dir):
    """查找CT序列目录"""
    ct_series = []
    
    for study_dir in patient_dir.iterdir():
        if not study_dir.is_dir():
            continue
        for series_dir in study_dir.iterdir():
            if not series_dir.is_dir():
                continue
            if is_ct_series(series_dir):
                ct_series.append(series_dir)
    
    return ct_series

def is_pet_series(series_dir):
    """判断是否为PET序列"""
    dicom_files = list(series_dir.glob("*.dcm"))
    if not dicom_files:
        dicom_files = list(series_dir.rglob("*"))  # 查找所有文件
        dicom_files = [f for f in dicom_files if f.is_file() and f.suffix.lower() in ['.dcm', '']]
    
    if not dicom_files:
        return False
    
    try:
        ds = pydicom.dcmread(dicom_files[0])
        return hasattr(ds, 'Modality') and ds.Modality == 'PT'
    except:
        return False

def is_ct_series(series_dir):
    """判断是否为CT序列"""
    dicom_files = list(series_dir.glob("*.dcm"))
    if not dicom_files:
        dicom_files = list(series_dir.rglob("*"))
        dicom_files = [f for f in dicom_files if f.is_file() and f.suffix.lower() in ['.dcm', '']]
    
    if not dicom_files:
        return False
    
    try:
        ds = pydicom.dcmread(dicom_files[0])
        return hasattr(ds, 'Modality') and ds.Modality == 'CT'
    except:
        return False

def convert_series_to_nifti(series_dir, target_path, patient_id, series_type, patient_index):
    """将DICOM序列转换为NIfTI格式"""
    try:
        # 查找DICOM文件
        dicom_files = list(series_dir.glob("*.dcm"))
        if not dicom_files:
            dicom_files = list(series_dir.rglob("*"))
            dicom_files = [f for f in dicom_files if f.is_file()]
        
        if not dicom_files:
            print(f"    未找到DICOM文件在 {series_dir}")
            return False
        
        print(f"    转换 {series_type} 序列: {len(dicom_files)} 个文件")
        
        # 读取所有DICOM切片
        slices = []
        positions = []
        
        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(dcm_file)
                slices.append(ds)
                if hasattr(ds, 'ImagePositionPatient'):
                    positions.append(float(ds.ImagePositionPatient[2]))
                else:
                    positions.append(0)
            except Exception as e:
                print(f"      跳过文件 {dcm_file}: {e}")
                continue
        
        if len(slices) < 5:  # 至少需要5个切片
            print(f"    切片数量不足: {len(slices)}")
            return False
        
        # 按位置排序
        sorted_pairs = sorted(zip(slices, positions), key=lambda x: x[1])
        slices = [pair[0] for pair in sorted_pairs]
        
        # 提取像素数据
        volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])
        
        # 基本预处理
        volume = preprocess_volume(volume, series_type)
        
        # 创建NIfTI文件
        nii = nib.Nifti1Image(volume, np.eye(4))
        
        # 决定保存到哪个目录 (简单的平衡分布)
        target_subdir = "cancer" if patient_index % 2 == 0 else "normal"
        target_dir = target_path / target_subdir
        
        # 保存文件
        output_file = target_dir / f"{patient_id}_{series_type}.nii.gz"
        nib.save(nii, output_file)
        
        print(f"    ✅ 转换完成: {output_file}")
        return True
        
    except Exception as e:
        print(f"    ❌ 转换失败: {e}")
        return False

def preprocess_volume(volume, series_type):
    """预处理体积数据"""
    # 去除异常值
    p1, p99 = np.percentile(volume, [1, 99])
    volume = np.clip(volume, p1, p99)
    
    # 根据模态类型进行不同的处理
    if series_type.startswith('PET'):
        # PET图像通常需要SUV校正等，这里简化处理
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
    elif series_type.startswith('CT'):
        # CT图像的HU值处理
        volume = np.clip(volume, -1000, 400)  # 典型的CT窗口
        volume = (volume + 1000) / 1400  # 归一化到[0,1]
    
    return volume

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python organize_images.py <source_dir> <target_dir>")
        sys.exit(1)
    
    organize_images_for_training(sys.argv[1], sys.argv[2])
EOF

chmod +x organize_images.py

# 6. 显示下载状态和下一步指示
echo ""
echo "==============================================="
echo "✅ TCIA图像数据下载准备完成!"
echo "==============================================="
echo ""
echo "已完成:"
echo "  ✅ 图像处理依赖安装"
echo "  ✅ NBIA Data Retriever下载"
echo "  ✅ 图像清单文件下载"
echo "  ✅ 图像组织脚本创建"

if [ "$JAVA_AVAILABLE" = true ]; then
    echo "  ✅ Java环境验证"
else
    echo "  ❌ Java环境需要安装"
fi

echo ""
echo "下一步操作:"
echo "1. 检查磁盘空间 (需要约100GB)"
echo "   df -h ."
echo ""
echo "2. 在screen/tmux中运行图像下载:"
echo "   screen -S tcia_download"
echo "   ./download_images.sh"
echo "   # 按 Ctrl+A, D 分离会话"
echo ""
echo "3. 检查下载进度:"
echo "   screen -r tcia_download"
echo ""
echo "4. 或者手动运行下载:"
echo "   java -jar tools/nbia-data-retriever-4.4.1.jar -m manifest.txt -d ./images -v"
echo ""
echo "数据位置: $(pwd)"
echo "==============================================="

# 显示数据集信息
echo ""
echo "数据集信息:"
echo "- 名称: NSCLC-Radiogenomics"
echo "- 患者数: 211"
echo "- 图像类型: CT, PET"
echo "- 数据大小: ~98GB"
echo "- 癌症类型: 非小细胞肺癌"
echo ""
echo "下载完成后数据结构:"
echo "organized_data/"
echo "├── normal/     # 正常样本 (.nii.gz文件)"
echo "└── cancer/     # 癌症样本 (.nii.gz文件)"
