#!/usr/bin/env python3
"""
記憶體診斷工具
用於分析 Breeze-ASR-25 微調的記憶體使用情況
"""

import gc
import os
from pathlib import Path

import numpy as np
import psutil
import torch


def print_memory_status():
    """顯示當前記憶體狀態"""
    print("=" * 60)
    print("🔍 記憶體狀態診斷")
    print("=" * 60)

    # 系統記憶體
    memory = psutil.virtual_memory()
    print(f"💾 系統記憶體:")
    print(f"   總計: {memory.total / 1024**3:.1f} GB")
    print(f"   已用: {memory.used / 1024**3:.1f} GB")
    print(f"   可用: {memory.available / 1024**3:.1f} GB")
    print(f"   使用率: {memory.percent:.1f}%")

    # GPU 記憶體
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = props.total_memory / 1024**3

            print(f"\n🔥 GPU {i} ({props.name}):")
            print(f"   總計: {total:.1f} GB")
            print(f"   已分配: {allocated:.2f} GB")
            print(f"   已保留: {reserved:.2f} GB")
            print(f"   可用: {total - reserved:.2f} GB")
            print(f"   使用率: {reserved/total*100:.1f}%")

            # 警告
            if reserved > total * 0.9:
                print(f"   ⚠️  警告: GPU 記憶體使用率過高!")
            elif reserved > total * 0.7:
                print(f"   ⚠️  注意: GPU 記憶體使用率較高")
            else:
                print(f"   ✅ GPU 記憶體狀態良好")
    else:
        print("\n❌ 未檢測到 GPU")

    print("=" * 60)


def estimate_model_memory():
    """估算 Breeze-ASR-25 模型記憶體需求"""
    print("\n📊 Breeze-ASR-25 記憶體需求估算")
    print("-" * 40)

    # 模型參數數量 (Breeze-ASR-25 約 244M 參數)
    model_params = 244_000_000

    # FP32 記憶體需求
    fp32_memory = model_params * 4 / 1024**3  # 4 bytes per parameter
    print(f"模型權重 (FP32): {fp32_memory:.2f} GB")

    # 梯度記憶體
    gradient_memory = fp32_memory
    print(f"梯度 (FP32): {gradient_memory:.2f} GB")

    # AdamW 優化器狀態 (2x 參數數量)
    optimizer_memory = model_params * 8 / 1024**3  # 8 bytes per parameter
    print(f"優化器狀態 (AdamW): {optimizer_memory:.2f} GB")

    # 激活值估算 (根據批次大小)
    batch_sizes = [1, 2, 4, 8]
    for batch_size in batch_sizes:
        # 粗略估算: 批次大小 * 序列長度 * 隱藏維度 * 層數
        activation_memory = batch_size * 3000 * 1024 * 24 / 1024**3  # 粗略估算
        print(f"激活值 (batch_size={batch_size}): {activation_memory:.2f} GB")

    # 總計
    base_memory = fp32_memory + gradient_memory + optimizer_memory
    print(f"\n基礎記憶體需求: {base_memory:.2f} GB")
    print(f"加上激活值 (batch_size=1): {base_memory + 0.7:.2f} GB")
    print(f"加上激活值 (batch_size=4): {base_memory + 2.8:.2f} GB")

    return base_memory


def test_memory_allocation():
    """測試記憶體分配"""
    print("\n🧪 記憶體分配測試")
    print("-" * 30)

    if not torch.cuda.is_available():
        print("❌ 無法測試 GPU 記憶體分配")
        return

    # 清理記憶體
    torch.cuda.empty_cache()
    gc.collect()

    # 測試不同大小的張量分配
    sizes_mb = [100, 500, 1000, 2000, 5000]

    for size_mb in sizes_mb:
        try:
            # 計算元素數量
            elements = size_mb * 1024 * 1024 // 4  # 4 bytes per float32

            # 嘗試分配
            tensor = torch.zeros(elements, dtype=torch.float32, device="cuda")

            # 檢查實際分配
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"✅ 成功分配 {size_mb} MB 張量 (實際: {allocated:.2f} GB)")

            # 清理
            del tensor
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"❌ 無法分配 {size_mb} MB 張量")
            break


def recommend_configuration():
    """根據記憶體狀態推薦配置"""
    print("\n🎯 配置推薦")
    print("-" * 20)

    if not torch.cuda.is_available():
        print("❌ 無 GPU，建議使用 CPU 模式")
        return

    # 獲取 GPU 記憶體
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_name = torch.cuda.get_device_name(0)

    print(f"GPU: {gpu_name}")
    print(f"記憶體: {total_memory:.1f} GB")

    if total_memory >= 40:
        print("✅ A100 配置:")
        print("   batch_size: 4")
        print("   gradient_accumulation_steps: 4")
        print("   gradient_checkpointing: False")
        print("   max_steps: 4000")

    elif total_memory >= 30:
        print("✅ V100 配置:")
        print("   batch_size: 2")
        print("   gradient_accumulation_steps: 8")
        print("   gradient_checkpointing: False")
        print("   max_steps: 2000")

    elif total_memory >= 20:
        print("⚠️  T4/P100 配置:")
        print("   batch_size: 1")
        print("   gradient_accumulation_steps: 16")
        print("   gradient_checkpointing: True")
        print("   max_steps: 1000")

    else:
        print("❌ 記憶體不足，建議:")
        print("   1. 升級到更大記憶體的 GPU")
        print("   2. 使用極致記憶體最佳化配置")
        print("   3. 考慮使用 CPU 模式")


def cleanup_memory():
    """清理記憶體"""
    print("\n🧹 記憶體清理")
    print("-" * 20)

    # 清理 Python 記憶體
    gc.collect()

    # 清理 CUDA 記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 重設統計
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except:
            pass

    print("✅ 記憶體清理完成")


def main():
    """主診斷流程"""
    print("🔍 Breeze-ASR-25 記憶體診斷工具")
    print("=" * 60)

    # 顯示當前狀態
    print_memory_status()

    # 估算模型需求
    estimate_model_memory()

    # 測試記憶體分配
    test_memory_allocation()

    # 推薦配置
    recommend_configuration()

    # 清理記憶體
    cleanup_memory()

    # 最終狀態
    print("\n📊 清理後狀態:")
    print_memory_status()


if __name__ == "__main__":
    main()
