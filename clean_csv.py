import pandas as pd


def clean_csv_file(csv_path):
    """
    從 CSV 檔案中移除「檔案位置」欄位為空的所有資料列。

    Args:
        csv_path (str): 要清理的 CSV 檔案路徑。
    """
    try:
        df = pd.read_csv(csv_path)

        if "檔案位置" not in df.columns:
            print(f"錯誤：在 {csv_path} 中找不到 '檔案位置' 欄位。")
            return

        original_rows = len(df)
        print(f"讀取 {csv_path}，原始資料共有 {original_rows} 行。")

        # 刪除 '檔案位置' 欄位為空值的列
        df.dropna(subset=["檔案位置"], inplace=True)

        cleaned_rows = len(df)
        removed_rows = original_rows - cleaned_rows

        # 將清理後的 DataFrame 儲存回 CSV
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print("檔案清理完成。")
        print(f"移除了 {removed_rows} 行資料。")
        print(f"檔案現在共有 {cleaned_rows} 行可供訓練。")

    except FileNotFoundError:
        print(f"錯誤：找不到 CSV 檔案於 {csv_path}")
    except Exception as e:
        print(f"處理過程中發生錯誤：{e}")


if __name__ == "__main__":
    # 根據您先前的操作，目標檔案設定為 'audio_2.csv'
    # 如果您想處理 'audio_1.csv'，請修改下方的檔名
    CSV_FILE_TO_CLEAN = "audio_1_converted.csv"
    clean_csv_file(CSV_FILE_TO_CLEAN)
