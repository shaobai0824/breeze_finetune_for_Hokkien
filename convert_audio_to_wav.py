# ==============================================================================
# 檔案：convert_audio_to_wav.py
# 描述：將 m4a 檔案轉換為 wav 格式，以便進行語音辨識評估
# ==============================================================================

import os
from pathlib import Path

import pandas as pd


def convert_m4a_to_wav_using_librosa(
    input_folder="debug_audio", output_folder="debug_audio_wav"
):
    """
    使用 librosa 將 m4a 檔案轉換為 wav 格式
    這個方法不需要 FFmpeg
    """
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        print("❌ 需要安裝 librosa 和 soundfile:")
        print("pip install librosa soundfile")
        return False

    # 創建輸出資料夾
    os.makedirs(output_folder, exist_ok=True)

    converted_files = []
    failed_files = []

    print(f"🔄 開始轉換音訊檔案...")
    print(f"📁 輸入資料夾: {input_folder}")
    print(f"📁 輸出資料夾: {output_folder}")
    print()

    # 尋找所有 m4a 檔案
    input_path = Path(input_folder)
    m4a_files = list(input_path.glob("*.m4a"))

    if not m4a_files:
        print("❌ 在輸入資料夾中找不到 .m4a 檔案")
        return False

    print(f"📊 找到 {len(m4a_files)} 個 .m4a 檔案")

    for m4a_file in m4a_files:
        try:
            # 讀取音訊檔案
            audio, sr = librosa.load(str(m4a_file), sr=16000, mono=True)

            # 生成輸出檔案名
            output_filename = m4a_file.stem + ".wav"
            output_path = Path(output_folder) / output_filename

            # 儲存為 wav 檔案
            sf.write(str(output_path), audio, sr)

            converted_files.append(
                {
                    "original": m4a_file.name,
                    "converted": output_filename,
                    "duration": len(audio) / sr,
                }
            )

            print(f"✅ {m4a_file.name} -> {output_filename} ({len(audio)/sr:.2f}s)")

        except Exception as e:
            failed_files.append({"file": m4a_file.name, "error": str(e)})
            print(f"❌ {m4a_file.name}: {e}")

    print()
    print(f"🎉 轉換完成!")
    print(f"   成功: {len(converted_files)} 個檔案")
    print(f"   失敗: {len(failed_files)} 個檔案")

    return converted_files, failed_files


def update_csv_for_wav_files(
    csv_path="test_references_example.csv",
    output_csv="test_references_wav.csv",
    converted_files=None,
):
    """
    更新 CSV 檔案中的檔案名稱，從 .m4a 改為 .wav
    """
    try:
        # 讀取原始 CSV
        df = pd.read_csv(csv_path)

        # 建立轉換對照表
        if converted_files:
            conversion_map = {
                item["original"]: item["converted"] for item in converted_files
            }
        else:
            # 如果沒有提供轉換清單，自動建立對照表
            conversion_map = {}
            for _, row in df.iterrows():
                if row["filename"].endswith(".m4a"):
                    wav_name = row["filename"].replace(".m4a", ".wav")
                    conversion_map[row["filename"]] = wav_name

        # 更新檔案名稱
        df["filename"] = df["filename"].map(lambda x: conversion_map.get(x, x))

        # 儲存新的 CSV
        df.to_csv(output_csv, index=False, encoding="utf-8")

        print(f"📄 已更新 CSV 檔案: {output_csv}")
        print(f"   更新了 {len(conversion_map)} 個檔案名稱")

        return output_csv

    except Exception as e:
        print(f"❌ 更新 CSV 檔案時發生錯誤: {e}")
        return None


def main():
    """主函數"""
    print("=== 音訊檔案格式轉換工具 ===\n")

    # 檢查資料夾是否存在
    if not os.path.exists("debug_audio"):
        print("❌ debug_audio 資料夾不存在")
        return

    print("✅ debug_audio 資料夾存在")

    # 轉換音訊檔案
    result = convert_m4a_to_wav_using_librosa()

    if result:
        converted_files, failed_files = result

        if converted_files:
            # 更新 CSV 檔案
            new_csv = update_csv_for_wav_files(converted_files=converted_files)

            if new_csv:
                print(f"\n🎯 下一步:")
                print(f"   1. 使用新的 CSV 檔案: {new_csv}")
                print(f"   2. 修改 evaluate_test_dataset.py 中的設定:")
                print(f"      TEST_CSV = '{new_csv}'")
                print(f"      TEST_AUDIO_FOLDER = 'debug_audio_wav'")
                print(f"   3. 重新執行評估: python evaluate_test_dataset.py")
        else:
            print("\n❌ 沒有成功轉換任何檔案")
    else:
        print("\n❌ 轉換失敗，請檢查是否已安裝必要套件:")
        print("pip install librosa soundfile")


if __name__ == "__main__":
    main()
