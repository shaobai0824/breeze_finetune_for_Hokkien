# ==============================================================================
# 檔案：evaluate_model.py
# 描述：批量評估一個資料夾內所有 .wav 檔案，並計算總體的 WER。
# ==============================================================================
import torch
import pandas as pd
from datasets import Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
import os
from tqdm import tqdm # 匯入 tqdm 來顯示進度條

def evaluate_folder(model_path: str, test_folder_path: str, reference_csv_path: str, language: str = "zh"):
    """
    對一個資料夾內的所有 .wav 檔案進行批量評估。

    Args:
        model_path (str): 您訓練好的模型資料夾路徑。
        test_folder_path (str): 包含測試 .wav 檔案的資料夾路徑。
        reference_csv_path (str): 包含檔名和對應文本的 CSV 檔案路徑。
    """
    # --- 1. 初始化模型和環境 ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"將使用設備：{device}")
    
    print(f"\n--- 正在載入您的微調模型自：{model_path} ---")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    
    # --- 2. 準備參考答案 ---
    print(f"\n--- 正在從 {reference_csv_path} 讀取參考文本 ---")
    reference_df = pd.read_csv(reference_csv_path)
    # 建立一個從檔名到文本的快速查詢字典，例如: {"common_voice_xx.wav": "你好"}
    # 我們只取檔案名稱，而不是完整路徑
    reference_dict = pd.Series(reference_df.transcription.values, index=reference_df.file.apply(lambda x: os.path.basename(x))).to_dict()

    # --- 3. 收集測試檔案並進行評估 ---
    predictions = []
    references = []
    
    # 找出所有 .wav 檔案
    test_files = [f for f in os.listdir(test_folder_path) if f.endswith('.wav')]
    if not test_files:
        print(f"錯誤：在資料夾 {test_folder_path} 中找不到任何 .wav 檔案。")
        return

    print(f"\n--- 找到 {len(test_files)} 個 .wav 檔案，開始進行批量辨識 ---")
    
    audio_loader = Audio(sampling_rate=16000)
    
    # 使用 tqdm 來建立一個進度條
    for filename in tqdm(test_files, desc="評估進度"):
        # 檢查此檔案是否有對應的參考文本
        if filename not in reference_dict:
            # print(f"警告：在 CSV 中找不到檔案 {filename} 的參考文本，將跳過此檔案。")
            continue

        file_path = os.path.join(test_folder_path, filename)
        
        try:
            # 載入並處理音訊
            audio_input = audio_loader.decode_example(audio_loader.encode_example(file_path))
            input_features = processor(
                audio_input["array"], 
                sampling_rate=audio_input["sampling_rate"], 
                return_tensors="pt"
            ).input_features.to(device)

            # 進行預測
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # 收集結果
            predictions.append(transcription)
            references.append(reference_dict[filename])

        except Exception as e:
            print(f"處理檔案 {filename} 時發生錯誤: {e}")

    # --- 4. 計算並顯示最終的 WER ---
    if not predictions:
        print("\n錯誤：沒有任何檔案被成功處理，無法計算 WER。")
        return
        
    print("\n--- 所有檔案辨識完畢，正在計算總體 WER ---")
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references) * 100
    
    print("\n================== 最終評估結果 ==================")
    print(f"總共評估了 {len(predictions)} 個檔案。")
    print(f"模型的字詞錯誤率 (WER): {wer:.2f}%")
    print("======================================================")

if __name__ == '__main__':
    # --- 請修改以下參數 ---
    
    # 您訓練好的模型的本地資料夾路徑
    MY_FINE_TUNED_MODEL_PATH = "./whisper-small-zh-finetune-final"
    
    # 包含您要測試的所有 .wav 檔案的資料夾路徑
    # 假設您從原始資料集中，分出了一個專門的 "test_wavs" 資料夾
    TEST_FOLDER_PATH = "path/to/your/test_wavs" 
    
    # 包含所有檔名和對應文本的原始 CSV 檔案
    REFERENCE_CSV = "output/final_audio_paths.csv"

    evaluate_folder(
        model_path=MY_FINE_TUNED_MODEL_PATH,
        test_folder_path=TEST_FOLDER_PATH,
        reference_csv_path=REFERENCE_CSV
    )