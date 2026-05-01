# -*- coding: utf-8 -*-
"""
数据加载与预处理模块
包含：TIFF 读取、mask 处理、按训练集拟合归一化、滑动窗口数据集、图连通性矩阵构建
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from gccl_model import build_k_hop_adj_matrix


def load_and_stack_tiffs(folder_path, prefix="Hubei_SMAP"):
    tif_files = [f for f in os.listdir(folder_path) if f.endswith(".tif") and f.startswith(prefix)]
    tif_files.sort()

    if not tif_files:
        raise ValueError(f"未找到以 {prefix} 开头的 TIFF 文件")

    print(f"找到 {len(tif_files)} 个 {prefix} 文件")
    all_data = []
    num_features = 5 if "ERA5" in prefix else 1

    for file in tif_files:
        file_path = os.path.join(folder_path, file)
        with rasterio.open(file_path) as src:
            data = src.read().astype(np.float32, copy=False)

        data = np.nan_to_num(data, nan=0.0)
        data[data < -9000] = 0.0

        bands, height, width = data.shape
        if bands % num_features != 0:
            raise ValueError(f"{file} 的 band 数 {bands} 不能被通道数 {num_features} 整除")

        days = bands // num_features
        data = data.reshape(days, num_features, height, width)
        all_data.append(data)

        if len(all_data) == 1:
            print(f"  单文件形状: {data.shape}, 数据类型: {data.dtype}")

    stacked_data = np.concatenate(all_data, axis=0)
    print(f"  堆叠后形状: {stacked_data.shape}")
    return stacked_data


def load_valid_mask(dataset_path, mask_name="Hubei_Mask.tif"):
    mask_path = os.path.join(dataset_path, mask_name)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    valid_mask = mask > 0
    print(f"加载空间 mask: {mask_name}, 有效像素 {int(valid_mask.sum())}/{valid_mask.size}")
    return valid_mask


def fit_min_max(data, valid_mask, per_channel=False):
    masked_values = data[:, :, valid_mask]

    if per_channel:
        data_min = masked_values.min(axis=(0, 2), keepdims=True).reshape(1, data.shape[1], 1, 1)
        data_max = masked_values.max(axis=(0, 2), keepdims=True).reshape(1, data.shape[1], 1, 1)
    else:
        data_min = float(masked_values.min())
        data_max = float(masked_values.max())

    return data_min, data_max


def normalize_with_params(data, data_min, data_max):
    return (data - data_min) / (data_max - data_min + 1e-8)


def min_max_normalize(data, valid_mask, per_channel=False):
    data_min, data_max = fit_min_max(data, valid_mask, per_channel=per_channel)
    normalized_data = normalize_with_params(data, data_min, data_max)
    return normalized_data, data_min, data_max


def apply_spatial_mask(data, valid_mask, fill_value=0.0):
    masked = data.copy()
    masked[:, :, ~valid_mask] = fill_value
    return masked


def denormalize(normalized_data, data_min, data_max):
    return normalized_data * (data_max - data_min + 1e-8) + data_min


class SpatiotemporalDataset(Dataset):
    def __init__(self, smap_data, era5_data=None, in_steps=3, out_steps=7, transform=None, is_train=True):
        self.smap_data = torch.from_numpy(smap_data).float()
        self.era5_data = torch.from_numpy(era5_data).float() if era5_data is not None else None
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.transform = transform
        self.is_train = is_train

        self.total_steps = in_steps + out_steps
        self.num_samples = len(smap_data) - self.total_steps + 1

        if self.num_samples <= 0:
            raise ValueError("数据长度不足，无法构造滑动窗口样本")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_start = idx
        input_end = idx + self.in_steps
        output_start = input_end
        output_end = output_start + self.out_steps

        smap_inputs = self.smap_data[input_start:input_end]
        smap_targets = self.smap_data[output_start:output_end]

        if self.era5_data is not None:
            era5_inputs = self.era5_data[input_start:input_end]
            inputs = torch.cat([smap_inputs, era5_inputs], dim=1)
        else:
            inputs = smap_inputs

        targets = smap_targets

        inputs = inputs.permute(0, 2, 3, 1)
        targets = targets.permute(0, 2, 3, 1)

        if self.transform:
            inputs = self.transform(inputs)
            targets = self.transform(targets)

        return inputs, targets


def split_data_by_time(data, train_ratio=0.7, val_ratio=0.1):
    total_steps = data.shape[0]
    train_end = int(total_steps * train_ratio)
    val_end = int(total_steps * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print(f"数据集划分 (总时间步: {total_steps}):")
    print(f"  训练集: 0 ~ {train_end} ({len(train_data)})")
    print(f"  验证集: {train_end} ~ {val_end} ({len(val_data)})")
    print(f"  测试集: {val_end} ~ {total_steps} ({len(test_data)})")

    return train_data, val_data, test_data


def prepare_dataloaders(dataset_path, in_steps=3, out_steps=7, batch_size=64, use_era5=False, theta=7, k_hop=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("加载空间 mask...")
    valid_mask = load_valid_mask(dataset_path)

    print("\n加载 SMAP 土壤水分数据...")
    smap_data = load_and_stack_tiffs(dataset_path, prefix="Hubei_SMAP")

    era5_data = None
    if use_era5:
        print("\n加载 ERA5 气象数据...")
        era5_data = load_and_stack_tiffs(dataset_path, prefix="Hubei_ERA5")

    print("\n先划分数据，再拟合归一化参数...")
    smap_train_raw, smap_val_raw, smap_test_raw = split_data_by_time(smap_data, 0.7, 0.1)

    smap_min, smap_max = fit_min_max(smap_train_raw, valid_mask, per_channel=False)
    smap_train = apply_spatial_mask(normalize_with_params(smap_train_raw, smap_min, smap_max), valid_mask)
    smap_val = apply_spatial_mask(normalize_with_params(smap_val_raw, smap_min, smap_max), valid_mask)
    smap_test = apply_spatial_mask(normalize_with_params(smap_test_raw, smap_min, smap_max), valid_mask)

    norm_params = {
        "smap_min": smap_min,
        "smap_max": smap_max,
        "valid_mask": valid_mask.astype(np.uint8),
    }

    if era5_data is not None:
        era5_train_raw, era5_val_raw, era5_test_raw = split_data_by_time(era5_data, 0.7, 0.1)
        era5_min, era5_max = fit_min_max(era5_train_raw, valid_mask, per_channel=True)

        era5_train = apply_spatial_mask(normalize_with_params(era5_train_raw, era5_min, era5_max), valid_mask)
        era5_val = apply_spatial_mask(normalize_with_params(era5_val_raw, era5_min, era5_max), valid_mask)
        era5_test = apply_spatial_mask(normalize_with_params(era5_test_raw, era5_min, era5_max), valid_mask)

        norm_params.update({"era5_min": era5_min, "era5_max": era5_max})
    else:
        era5_train = era5_val = era5_test = None

    print("\n构建图连通性矩阵...")
    smap_train_for_graph = torch.from_numpy(smap_train).float().squeeze(1)
    graph_mask = torch.from_numpy(valid_mask)
    adj_matrix = build_k_hop_adj_matrix(smap_train_for_graph, theta=theta, k=k_hop, valid_mask=graph_mask)
    print(f"  邻接矩阵形状: {adj_matrix.shape}")
    adj_matrix = adj_matrix.to(device)

    print("\n创建 Dataset...")
    train_dataset = SpatiotemporalDataset(smap_train, era5_train, in_steps, out_steps, is_train=True)
    val_dataset = SpatiotemporalDataset(smap_val, era5_val, in_steps, out_steps, is_train=False)
    test_dataset = SpatiotemporalDataset(smap_test, era5_test, in_steps, out_steps, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("\n" + "=" * 50)
    print("数据准备完成!")

    return train_loader, val_loader, test_loader, adj_matrix, norm_params
