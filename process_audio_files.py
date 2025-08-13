import os

import pandas as pd


def find_audio_files(csv_path, folders):
    """
    Finds audio files listed in a CSV column within specified folders and adds their paths to the CSV.

    Args:
        csv_path (str): The path to the input CSV file.
        folders (list): A list of folder paths to search for audio files.
    """
    try:
        # 讀取 CSV 檔案
        df = pd.read_csv(csv_path)
        print(f"成功讀取 {csv_path}，共有 {len(df)} 行資料。")

        # 建立一個集合來儲存所有音檔的相對路徑
        audio_files_map = {}
        for folder in folders:
            for root, _, files in os.walk(folder):
                for file in files:
                    filename, _ = os.path.splitext(file)
                    audio_files_map[filename] = os.path.join(root, file)

        print(f"在指定資料夾中總共找到 {len(audio_files_map)} 個音檔。")

        # 根據 "羅馬字音檔檔名" 尋找檔案
        def find_path(row):
            filename_to_find = row["羅馬字音檔檔名"]
            return audio_files_map.get(filename_to_find)

        df["檔案位置"] = df.apply(find_path, axis=1)

        # 將更新後的 DataFrame 儲存回 CSV
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"成功更新 {csv_path}，已新增 '檔案位置' 欄位。")

    except FileNotFoundError:
        print(f"錯誤：找不到 CSV 檔案於 {csv_path}")
    except Exception as e:
        print(f"處理過程中發生錯誤：{e}")


if __name__ == "__main__":
    CSV_FILE = "audio_1_converted.csv"
    SEARCH_FOLDERS = [
        "leku-wav-20250728T095826Z-1-001",
        "sutiau-wav-20250728T095826Z-1-001",
    ]
    find_audio_files(CSV_FILE, SEARCH_FOLDERS)
