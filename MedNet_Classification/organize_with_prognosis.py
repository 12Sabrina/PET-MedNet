#!/usr/bin/env python3
"""
基于临床预后数据的TCIA PET数据组织脚本
根据预后时间列确定良性/恶性标签

⚠️ 重要：本脚本仅处理PET图像数据，忽略CT和其他模态
- 输入：TCIA DICOM数据
- 输出：良性/恶性分类的PET NIfTI文件
- 标签：基于'Time to Death (days)'列
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import pydicom
import nibabel as nib

def analyze_clinical_data(clinical_file):
    """分析临床数据，确定标签分布"""
    if not os.path.exists(clinical_file):
        print(f"❌ 未找到临床数据文件: {clinical_file}")
        return None, None, None
    
    try:
        # 读取临床数据
        clinical_data = pd.read_csv(clinical_file)
        print(f"✅ 成功读取临床数据: {len(clinical_data)} 个患者")
        
        # 显示所有列名
        print("\n可用的列:")
        for i, col in enumerate(clinical_data.columns):
            print(f"  {i+1:2d}. {col}")
        
        # 查找可能的预后时间列
        prognosis_columns = []
        keywords = ['survival', 'prognosis', 'time', 'follow', 'outcome', 'days', 'months']
        
        for col in clinical_data.columns:
            for keyword in keywords:
                if keyword.lower() in col.lower():
                    prognosis_columns.append(col)
                    break
        
        if prognosis_columns:
            print(f"\n🔍 发现可能的预后时间列:")
            for i, col in enumerate(prognosis_columns):
                non_null_count = clinical_data[col].notna().sum()
                null_count = clinical_data[col].isna().sum()
                print(f"  {i+1}. {col}: {non_null_count} 有数据, {null_count} 为空")
        
        return clinical_data, prognosis_columns, clinical_data.columns.tolist()
        
    except Exception as e:
        print(f"❌ 读取临床数据失败: {e}")
        return None, None, None

def organize_images_with_prognosis(source_dir, target_dir, clinical_file, prognosis_column):
    """根据预后时间组织图像数据"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 创建目标目录
    benign_dir = target_path / "benign"    # 良性 (预后时间为空)
    malignant_dir = target_path / "malignant"  # 恶性 (预后时间有数字)
    benign_dir.mkdir(parents=True, exist_ok=True)
    malignant_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"根据预后时间列 '{prognosis_column}' 组织数据...")
    
    # 读取临床数据
    clinical_data = pd.read_csv(clinical_file)
    
    # 统计变量
    processed_count = 0
    benign_count = 0
    malignant_count = 0
    no_clinical_data = 0
    no_images = 0
    no_pet_data = 0  # 新增：没有PET数据的患者计数
    
    # 遍历患者目录
    for patient_dir in source_path.iterdir():
        if not patient_dir.is_dir():
            continue
            
        patient_id = patient_dir.name
        print(f"处理患者: {patient_id}")
        
        image_series = find_image_series(patient_dir)
        
        if not image_series:
            print(f"  ❌ 未找到PET图像序列")
            no_pet_data += 1
            continues += 1
            continue
        
        # 在临床数据中查找该患者
        # 尝试不同的患者ID匹配方式
        patient_info = None
        for id_col in ['Case ID', 'PatientID', 'Patient ID', 'ID', 'Subject ID']:
            if id_col in clinical_data.columns:
                patient_info = clinical_data[clinical_data[id_col] == patient_id]
                if not patient_info.empty:
                    break
                # 也尝试匹配去掉前缀的ID
                clean_id = patient_id.replace('R01-', '').replace('R01', '').lstrip('0')
                patient_info = clinical_data[clinical_data[id_col].astype(str).str.contains(clean_id, na=False)]
                if not patient_info.empty:
                    break
        
        if patient_info is None or patient_info.empty:
            print(f"  ⚠️  未找到患者 {patient_id} 的临床数据")
            no_clinical_data += 1
            continue
        
        # 检查预后时间列
        prognosis_value = patient_info[prognosis_column].iloc[0]
        
        if pd.isna(prognosis_value) or prognosis_value == '' or prognosis_value == 0:
            # 预后时间为空或0 -> 良性
            target_subdir = benign_dir
            label = "良性"
            benign_count += 1
        else:
            # 预后时间有数字 -> 恶性
            target_subdir = malignant_dir
            label = "恶性"
            malignant_count += 1
        
        print(f"  📋 预后时间: {prognosis_value} -> {label}")
        
        # 只转换并保存PET图像
        success = False
        pet_series = [s for s in image_series if s['type'] == 'PET']
        
        if pet_series:
            # 只处理第一个PET序列
            if convert_series_to_nifti(pet_series[0]['path'], target_subdir, patient_id, 'PET'):
                success = True
                print(f"  ✅ 成功处理PET序列")
    # 显示统计结果
    print(f"\n" + "="*50)
    print(f"TCIA PET数据组织完成统计:")
    print(f"="*50)
    print(f"总患者目录数: {len([d for d in source_path.iterdir() if d.is_dir()])}")
    print(f"成功处理: {processed_count}")
    print(f"良性PET样本: {benign_count}")
    print(f"恶性PET样本: {malignant_count}")
    print(f"无临床数据: {no_clinical_data}")
    print(f"无PET图像数据: {no_pet_data}")
    print(f"")
    print(f"最终PET数据分布:")
    print(f"  良性PET样本: {len(list(benign_dir.glob('*.nii.gz')))} 个")
    print(f"  恶性PET样本: {len(list(malignant_dir.glob('*.nii.gz')))} 个")
    print(f"良性肿瘤: {benign_count}")
    print(f"恶性肿瘤: {malignant_count}")
    print(f"无临床数据: {no_clinical_data}")
    print(f"无图像数据: {no_images}")
    print(f"")
    print(f"最终数据分布:")
    print(f"  良性样本: {len(list(benign_dir.glob('*.nii.gz')))} 个")
    print(f"  恶性样本: {len(list(malignant_dir.glob('*.nii.gz')))} 个")

def find_image_series(patient_dir):
    """查找PET图像序列 - 只处理PET数据"""
    series_list = []
    
    for study_dir in patient_dir.iterdir():
        if not study_dir.is_dir():
            continue
        for series_dir in study_dir.iterdir():
            if not series_dir.is_dir():
                continue
            
            # 只检查PET序列，忽略CT和其他模态
            if is_pet_series(series_dir):
                series_list.append({
                    'path': series_dir,
                    'type': 'PET',
                    'priority': 1
                })
    
    # 只返回PET序列
    return series_list

def is_pet_series(series_dir):
    """判断是否为PET序列"""
    dicom_files = list(series_dir.glob("*.dcm"))
    if not dicom_files:
        dicom_files = list(series_dir.rglob("*"))
        dicom_files = [f for f in dicom_files if f.is_file()]
    
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
        dicom_files = [f for f in dicom_files if f.is_file()]
    
    if not dicom_files:
        return False
    
    try:
        ds = pydicom.dcmread(dicom_files[0])
        return hasattr(ds, 'Modality') and ds.Modality == 'CT'
    except:
        return False

def convert_series_to_nifti(series_dir, target_dir, patient_id, series_type):
    """将DICOM序列转换为NIfTI格式"""
    try:
        # 查找DICOM文件
        dicom_files = list(series_dir.glob("*.dcm"))
        if not dicom_files:
            dicom_files = list(series_dir.rglob("*"))
            dicom_files = [f for f in dicom_files if f.is_file()]
        
        if not dicom_files:
            print(f"    ❌ 未找到DICOM文件")
            return False
        
        print(f"    🔄 转换 {series_type} 序列: {len(dicom_files)} 个文件")
        
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
                continue
        
        if len(slices) < 5:  # 至少需要5个切片
            print(f"    ❌ 切片数量不足: {len(slices)}")
            return False
        
        # 按位置排序
        sorted_pairs = sorted(zip(slices, positions), key=lambda x: x[1])
        slices = [pair[0] for pair in sorted_pairs]
        
        # 提取像素数据
        volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])
        
        # 预处理
        volume = preprocess_volume(volume, series_type)
        
        # 创建NIfTI文件
        nii = nib.Nifti1Image(volume, np.eye(4))
        
        # 保存文件
        output_file = target_dir / f"{patient_id}_{series_type}.nii.gz"
        nib.save(nii, output_file)
        
        print(f"    ✅ 保存: {output_file.name}")
        return True
        
    except Exception as e:
        print(f"    ❌ 转换失败: {e}")
        return False

def preprocess_volume(volume, series_type):
    """预处理体积数据"""
    # 去除异常值
    p1, p99 = np.percentile(volume, [1, 99])
    volume = np.clip(volume, p1, p99)
    
    # 根据模态类型进行处理
    if series_type == 'PET':
        # PET图像归一化
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
    elif series_type == 'CT':
        # CT图像HU值处理
        volume = np.clip(volume, -1000, 400)
        volume = (volume + 1000) / 1400
    
    return volume

def interactive_setup():
    """交互式设置"""
    print("=== TCIA数据标签分析和组织 ===\n")
    
    # 检查临床数据文件
    clinical_file = "clinical/clinical_data.csv"
    clinical_data, prognosis_columns, all_columns = analyze_clinical_data(clinical_file)
    
    if clinical_data is None:
        print("请先下载临床数据!")
        return False
    
    # 让用户选择预后时间列
    if prognosis_columns:
        print(f"\n请选择预后时间列:")
        for i, col in enumerate(prognosis_columns):
            print(f"  {i+1}. {col}")
        
        try:
            choice = int(input("\n请输入序号: ")) - 1
            if 0 <= choice < len(prognosis_columns):
                prognosis_column = prognosis_columns[choice]
            else:
                print("无效选择，使用第一个列")
                prognosis_column = prognosis_columns[0]
        except:
            prognosis_column = prognosis_columns[0]
    else:
        print(f"\n未找到明显的预后时间列，请手动选择:")
        for i, col in enumerate(all_columns):
            print(f"  {i+1:2d}. {col}")
        
        try:
            choice = int(input("\n请输入序号: ")) - 1
            if 0 <= choice < len(all_columns):
                prognosis_column = all_columns[choice]
            else:
                print("无效选择")
                return False
        except:
            print("输入错误")
            return False
    
    print(f"\n✅ 选择的预后时间列: {prognosis_column}")
    
    # 显示该列的数据分布
    print(f"\n数据分布预览:")
    value_counts = clinical_data[prognosis_column].value_counts(dropna=False)
    null_count = clinical_data[prognosis_column].isna().sum()
    non_null_count = clinical_data[prognosis_column].notna().sum()
    
    print(f"  空值 (良性): {null_count}")
    print(f"  有数值 (恶性): {non_null_count}")
    
    if non_null_count > 0:
        print(f"  数值范围: {clinical_data[prognosis_column].min():.1f} - {clinical_data[prognosis_column].max():.1f}")
    
    # 确认继续
    confirm = input(f"\n确认使用 '{prognosis_column}' 作为标签依据? [y/N]: ")
    if confirm.lower() != 'y':
        print("操作取消")
        return False
    
    return prognosis_column

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # 命令行模式
        source_dir = sys.argv[1]
        target_dir = sys.argv[2]
        
        # 交互式选择预后时间列
        prognosis_column = interactive_setup()
        if not prognosis_column:
            sys.exit(1)
        
        organize_images_with_prognosis(source_dir, target_dir, "clinical/clinical_data.csv", prognosis_column)
    
    elif len(sys.argv) == 4:
        # 指定预后时间列
        source_dir = sys.argv[1]
        target_dir = sys.argv[2]
        prognosis_column = sys.argv[3]
        
        organize_images_with_prognosis(source_dir, target_dir, "clinical/clinical_data.csv", prognosis_column)
    
    else:
        print("用法:")
        print("  python organize_with_prognosis.py <source_dir> <target_dir>  # 交互式选择列")
        print("  python organize_with_prognosis.py <source_dir> <target_dir> <prognosis_column>  # 指定列名")
        sys.exit(1)
