#!/usr/bin/env python3
"""
訓練監控腳本 - 監控 GPU、CPU 和記憶體使用情況
使用方法：在另一個終端中運行 python monitor_training.py
"""

import json
import subprocess
import time
from datetime import datetime

import psutil


def get_gpu_info():
    """獲取 GPU 資訊"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            gpu_data = result.stdout.strip().split(", ")
            return {
                "temperature": int(gpu_data[0]),
                "utilization": int(gpu_data[1]),
                "memory_used": int(gpu_data[2]),
                "memory_total": int(gpu_data[3]),
            }
    except Exception as e:
        print(f"無法獲取 GPU 資訊: {e}")
    return None


def get_system_info():
    """獲取系統資訊"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()

    return {
        "cpu_percent": cpu_percent,
        "memory_used_gb": memory.used / (1024**3),
        "memory_total_gb": memory.total / (1024**3),
        "memory_percent": memory.percent,
    }


def monitor_training(interval=10):
    """監控訓練過程"""
    print("開始監控訓練過程...")
    print("按 Ctrl+C 停止監控")
    print("-" * 80)

    try:
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S")

            # 獲取系統資訊
            sys_info = get_system_info()

            # 獲取 GPU 資訊
            gpu_info = get_gpu_info()

            # 顯示資訊
            print(f"\n[{timestamp}] 系統資源使用情況:")
            print(f"CPU 使用率: {sys_info['cpu_percent']:.1f}%")
            print(
                f"記憶體使用: {sys_info['memory_used_gb']:.1f}GB / {sys_info['memory_total_gb']:.1f}GB ({sys_info['memory_percent']:.1f}%)"
            )

            if gpu_info:
                gpu_memory_percent = (
                    gpu_info["memory_used"] / gpu_info["memory_total"]
                ) * 100
                print(f"GPU 使用率: {gpu_info['utilization']}%")
                print(
                    f"GPU 記憶體: {gpu_info['memory_used']}MB / {gpu_info['memory_total']}MB ({gpu_memory_percent:.1f}%)"
                )
                print(f"GPU 溫度: {gpu_info['temperature']}°C")

                # 警告檢查
                if gpu_info["temperature"] > 80:
                    print("⚠️  警告：GPU 溫度過高！")
                if gpu_memory_percent > 90:
                    print("⚠️  警告：GPU 記憶體使用率過高！")
            else:
                print("無法獲取 GPU 資訊")

            print("-" * 50)
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n監控已停止")


if __name__ == "__main__":
    monitor_training()
