import csv
import datetime
import os

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

# --- 設定區 (您可以修改這些值) ---
SAVE_DIRECTORY = "recordings"  # 儲存錄音檔的資料夾名稱
CSV_FILENAME = "範例語句_日常聊天用.csv"  # CSV 記錄檔的名稱
SAMPLE_RATE = 44100  # 取樣率 (Hz)，CD 音質為 44100
CHANNELS = 1  # 聲道數 (1: 單聲道, 2: 立體聲)
# ------------------------------------


def ensure_directory_exists(path):
    """確保指定的資料夾存在，如果不存在就建立它"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"已建立資料夾: {path}")


def append_to_csv(filepath, timestamp, description):
    """將檔案資訊附加到 CSV 檔案中"""
    # 取得檔案的絕對路徑
    absolute_path = os.path.abspath(filepath)
    file_exists = os.path.isfile(CSV_FILENAME)

    with open(CSV_FILENAME, mode="a", newline="", encoding="utf-8-sig") as csv_file:
        # 'utf-8-sig' 編碼可以讓 Excel 正確開啟中文，不會亂碼
        # newline='' 可以避免寫入時多出空白行

        writer = csv.writer(csv_file)

        # 如果檔案是新建立的，先寫入標頭 (Header)
        if not file_exists or os.path.getsize(CSV_FILENAME) == 0:
            writer.writerow(["檔案絕對路徑", "錄音時間", "描述"])

        # 寫入這筆錄音的資料
        writer.writerow([absolute_path, timestamp, description])
    print(f"記錄已儲存至: {CSV_FILENAME}")


def record_audio_and_save():
    """主功能：錄音並儲存路徑到 CSV"""
    # 確保儲存目錄存在
    ensure_directory_exists(SAVE_DIRECTORY)

    # 讓使用者輸入檔案描述
    description = input("請輸入這次錄音的描述 (可直接按 Enter 跳過): ")

    # 產生獨一無二的檔名
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp_str}.wav"
    filepath = os.path.join(SAVE_DIRECTORY, filename)

    print("\n" + "=" * 40)
    print(f"準備錄音... 檔名將是: {filename}")
    print("錄音即將開始，請對著麥克風說話。")
    print("按下 Ctrl+C 即可結束錄音。")
    print("=" * 40 + "\n")

    try:
        # 開始錄音
        recording = sd.rec(
            int(120 * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
        )
        # 設定一個較長的預設錄音時間 (例如 120 秒)，使用者可以透過 Ctrl+C 提早中斷
        sd.wait()  # 等待錄音自然結束或被中斷

    except KeyboardInterrupt:
        # 當使用者按下 Ctrl+C 時，這個區塊會被執行
        sd.stop()
        print("\n錄音已手動結束。")

    except Exception as e:
        print(f"發生錯誤：{e}")
        return

    # 儲存錄音檔
    print(f"正在儲存檔案至: {filepath}")
    write(filepath, SAMPLE_RATE, recording)

    # 將檔案資訊寫入 CSV
    append_to_csv(filepath, now.strftime("%Y-%m-%d %H:%M:%S"), description)

    print("\n流程完成！")


if __name__ == "__main__":
    record_audio_and_save()
