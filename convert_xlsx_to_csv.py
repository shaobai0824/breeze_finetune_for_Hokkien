import os

import pandas as pd


def convert_xlsx_to_csv(xlsx_path, csv_path):
    """
    將 XLSX 檔案轉換為 UTF-8 編碼的 CSV 檔案，以避免亂碼。

    Args:
        xlsx_path (str): 輸入的 XLSX 檔案路徑。
        csv_path (str): 輸出的 CSV 檔案路徑。
    """
    try:
        # 檢查 XLSX 檔案是否存在
        if not os.path.exists(xlsx_path):
            print(f"錯誤：找不到指定的 XLSX 檔案於 {xlsx_path}")
            return

        # 讀取 XLSX 檔案
        print(f"正在讀取檔案：{xlsx_path}")
        df = pd.read_excel(xlsx_path)

        # 將 DataFrame 寫入 CSV，使用 'utf-8-sig' 編碼
        # 'utf-8-sig' 會在檔案開頭加入 BOM (Byte Order Mark)，有助於 Excel 正確識別 UTF-8
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print(f"成功將檔案轉換並儲存至：{csv_path}")

    except Exception as e:
        print(f"轉換過程中發生錯誤：{e}")


if __name__ == "__main__":
    # --- 請在此處設定您要轉換的檔案 ---
    # 輸入的 XLSX 檔案
    XLSX_FILE = "audio_1.xlsx"

    # 輸出的 CSV 檔案名稱
    CSV_OUTPUT_PATH = "audio_1_converted.csv"
    # --- 設定結束 ---

    convert_xlsx_to_csv(XLSX_FILE, CSV_OUTPUT_PATH)
