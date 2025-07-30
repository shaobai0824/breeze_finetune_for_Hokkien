#!/usr/bin/env python3
"""
修改 CSV 檔案中的 file 欄位路徑
將路徑轉換為相對路徑 ./standard/ 格式
"""

import os
from pathlib import Path

import pandas as pd


def fix_audio_path_to_relative(path_str):
    """將音訊檔案路徑轉換為相對路徑格式"""
    if not isinstance(path_str, str):
        return path_str

    path_str = str(path_str).strip()

    # 如果已經是相對路徑格式，直接返回
    if path_str.startswith("./standard/"):
        return path_str

    # 處理 Google Drive 路徑
    if path_str.startswith("/content/drive/MyDrive/"):
        # 移除 /content/drive/MyDrive/ 前綴
        relative_part = path_str.replace("/content/drive/MyDrive/", "")
        # 轉換為 ./standard/ 格式
        if relative_part.startswith("standard/"):
            return f"./{relative_part}"
        else:
            return f"./standard/{relative_part}"

    # 處理 Windows 絕對路徑
    if path_str.startswith(("C:", "D:", "E:", "F:")):
        path_parts = Path(path_str).parts
        try:
            # 尋找 standard 目錄
            standard_idx = path_parts.index("standard")
            if standard_idx + 1 < len(path_parts):
                relative_path = "/".join(path_parts[standard_idx:])
                return f"./{relative_path}"
        except ValueError:
            pass

        # 如果找不到 standard，使用檔案名
        filename = Path(path_str).name
        return f"./standard/{filename}"

    # 處理其他路徑
    if not path_str.startswith("./"):
        return f"./standard/{path_str}"

    return path_str


def fix_csv_paths(csv_file, output_file=None):
    """修改 CSV 檔案中的 file 欄位路徑"""
    print(f"📊 處理 CSV 檔案: {csv_file}")

    # 檢查檔案是否存在
    if not Path(csv_file).exists():
        print(f"❌ 檔案不存在: {csv_file}")
        return False

    # 讀取 CSV
    df = pd.read_csv(csv_file)
    print(f"✅ 成功讀取 CSV，共 {len(df)} 筆資料")

    # 檢查是否有 file 欄位
    if "file" not in df.columns:
        print("❌ CSV 檔案中沒有 'file' 欄位")
        print(f"   現有欄位: {list(df.columns)}")
        return False

    # 顯示原始路徑範例
    print("\n📋 原始路徑範例:")
    for i, path in enumerate(df["file"].head(3)):
        print(f"   {i+1}. {path}")

    # 轉換路徑
    print("\n🔧 開始轉換路徑...")
    df["file"] = df["file"].apply(fix_audio_path_to_relative)

    # 顯示轉換後的路徑範例
    print("\n📋 轉換後路徑範例:")
    for i, path in enumerate(df["file"].head(3)):
        print(f"   {i+1}. {path}")

    # 驗證檔案是否存在
    print("\n🔍 驗證檔案路徑...")
    missing_files = []
    valid_files = 0

    for idx, row in df.iterrows():
        if Path(row["file"]).exists():
            valid_files += 1
        else:
            missing_files.append(row["file"])

    print(f"✅ 有效檔案: {valid_files}/{len(df)}")
    if missing_files:
        print(f"⚠️  缺失檔案: {len(missing_files)} 個")
        for file in missing_files[:5]:  # 只顯示前5個
            print(f"   - {file}")
        if len(missing_files) > 5:
            print(f"   ... 還有 {len(missing_files) - 5} 個檔案")

    # 決定輸出檔案名
    if output_file is None:
        base_name = Path(csv_file).stem
        output_file = f"{base_name}_fixed.csv"

    # 保存修改後的 CSV
    df.to_csv(output_file, index=False)
    print(f"\n💾 已保存修改後的 CSV: {output_file}")

    return True


def main():
    """主函數"""
    print("🚀 CSV 路徑修正工具")
    print("=" * 50)

    # 處理訓練資料
    train_csv = "metadata_train.csv"
    if Path(train_csv).exists():
        print(f"\n📝 處理訓練資料...")
        fix_csv_paths(train_csv, "metadata_train_fixed.csv")
    else:
        print(f"⚠️  找不到 {train_csv}")

    # 處理測試資料
    test_csv = "metadata_test.csv"
    if Path(test_csv).exists():
        print(f"\n📝 處理測試資料...")
        fix_csv_paths(test_csv, "metadata_test_fixed.csv")
    else:
        print(f"⚠️  找不到 {test_csv}")

    print("\n✅ 路徑修正完成！")
    print("\n📋 使用說明:")
    print("1. 使用修改後的 CSV 檔案進行訓練")
    print("2. 確保 ./standard/ 目錄下有對應的音訊檔案")
    print("3. 如果發現缺失檔案，請檢查檔案路徑是否正確")


if __name__ == "__main__":
    main()
