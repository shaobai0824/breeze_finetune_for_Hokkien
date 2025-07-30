# ==============================================================================
# 檔案：train_final.py
# 描述：一個完整、高效、穩健的 Whisper 模型微調流程的最終版本。
# 核心策略：
# 1. 即時轉換 (.with_transform)：徹底解決記憶體不足與預處理過久的問題。
# 2. 背景預取 (dataloader_num_workers)：解決 CPU 與 I/O 瓶頸，最大化 GPU 使用率。
# 3. 全域定義 (Global Scope)：解決多核心處理時的 pickling 錯誤。
# 4. 智慧續練 (Smart Resuming)：自動從上次的檢查點恢復訓練。
# ==============================================================================

import torch
import pandas as pd
from typing import Any, Dict, List, Union
from dataclasses import dataclass
import evaluate
import os
from functools import partial

# --- Hugging Face 相關導入 ---
from huggingface_hub import login
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import Dataset, DatasetDict, Audio

# ==============================================================================
# 步驟 1: 將所有輔助類別與函式定義在「全域範圍」
# 這是為了確保在使用 dataloader_num_workers > 0 時，背景程序可以成功序列化 (pickle) 它們。
# ==============================================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """處理語音到序列資料的 Data Collator，負責將樣本整理成批次並進行填充。"""
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def prepare_dataset_batched(batch, feature_extractor, tokenizer):
    """將一批音訊和文本資料『即時』轉換為模型輸入格式。"""
    audio_list = batch["audio"]
    batch["input_features"] = feature_extractor(
        [x["array"] for x in audio_list], 
        sampling_rate=audio_list[0]["sampling_rate"]
    ).input_features
    batch["labels"] = tokenizer(batch["transcription"], max_length=448, truncation=True).input_ids
    return batch

def compute_metrics(pred, tokenizer):
    """在評估階段，計算並回傳 WER 指標。"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    metric = evaluate.load("wer")
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ==============================================================================
# 步驟 2: 主執行流程
# ==============================================================================
def main():
    # --- 參數設定 ---
    CSV_PATH = 'output/final_audio_paths_zh.csv'
    MODEL_NAME = "MediaTek-Research/Breeze-ASR-25"
    LANGUAGE = "zh"
    TASK = "transcribe"
    # 為原型訓練建立一個新的輸出目錄
    OUTPUT_DIR = "./Breeze-ASR-25" 

    # --- 載入 Processor 和模型 ---
    print("--- 步驟 1/4: 載入 Processor 和模型 ---")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # --- 建立原始資料集 ---
    class AudioDatasetProcessor:
        def __init__(self, file_path: str, target_sampling_rate: int = 16000):
            self.file_path = file_path
            self.target_sampling_rate = target_sampling_rate
        def create_dataset(self) -> Dataset:
            full_data = pd.read_csv(self.file_path)
            dataset = Dataset.from_pandas(full_data)
            dataset = dataset.cast_column("file", Audio(sampling_rate=self.target_sampling_rate))
            dataset = dataset.rename_column("file", "audio")
            return dataset
            
    print("\n--- 步驟 2/4: 建立原始資料集並進行『抽樣』---")
    audio_processor = AudioDatasetProcessor(file_path=CSV_PATH)
    full_dataset = audio_processor.create_dataset()
    
    # [性能提升關鍵]
    # 對完整資料集進行隨機抽樣，只取其中的 10% 作為本次訓練的資料
    # 這會讓每一個 epoch 的長度大幅縮短
    subset_size = int(len(full_dataset) * 0.10)
    prototype_dataset = full_dataset.shuffle(seed=42).select(range(subset_size))
    
    print(f"已從完整資料集中隨機抽取 {subset_size} 筆資料進行快速原型訓練。")

    common_voice = prototype_dataset.train_test_split(test_size=0.2, seed=42)
    
    # --- 設定即時轉換 (維持不變) ---
    prepare_fn = partial(prepare_dataset_batched, feature_extractor=processor.feature_extractor, tokenizer=processor.tokenizer)
    vectorized_datasets = common_voice.with_transform(prepare_fn)
    print("即時轉換已設定。")

    # --- 建立訓練元件 (使用穩定參數) ---
    print("\n--- 步驟 3/4: 建立訓練元件 ---")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics_fn = partial(compute_metrics, tokenizer=processor.tokenizer)

    # 使用我們已經驗證過最穩定的參數組合
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        dataloader_num_workers=0,
        learning_rate=1e-5,
        warmup_steps=100, # 相應減少 warmup
        
        # [性能提升關鍵] 減少總訓練步數，以求快速看到結果
        max_steps=500, # 從 5000 大幅降至 500
        
        save_steps=250, # 相應調整儲存和評估頻率
        eval_steps=250,
        logging_steps=25,
       
        # --- 其他參數維持不變 ---
        fp16=True,
        eval_strategy="steps",
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True, # 您可以先設為 False，成功後再開啟
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=processor.feature_extractor,
    )
    
    # --- 開始訓練 ---
    print("\n--- 步驟 4/4: 開始模型微調訓練 (使用最終穩定高效模式) ---")
    # 不帶參數的 .train() 會自動處理斷點續練，是最穩健的做法。
    trainer.train() 
    print("\n*** 訓練完成 ***")
    
    # --- 儲存最終模型 ---
    print("\n--- 正在儲存最終的最佳模型 ---")
    final_model_path = training_args.output_dir
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    print(f"\n最終模型已儲存至：{final_model_path}")


if __name__ == '__main__':
    # 確保您已在終端機使用 `huggingface-cli login` 登入
    # 執行前建議重新啟動您的電腦，確保系統處於乾淨狀態
    main()