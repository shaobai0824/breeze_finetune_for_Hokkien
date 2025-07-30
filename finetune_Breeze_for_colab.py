#!/usr/bin/env python3
"""
Breeze-ASR-25 最終修正版本 - 基於 train.py 成功模式
使用正確的資料處理方式和訓練參數
"""

import gc
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import evaluate
import jiwer
import numpy as np
import pandas as pd
import torch
from datasets import Audio, Dataset, DatasetDict
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# ==============================================================================
# 基本配置
# ==============================================================================

MODEL_ID = "MediaTek-Research/Breeze-ASR-25"
TRAIN_CSV = "metadata_train.csv"
TEST_CSV = "metadata_test.csv"
OUTPUT_DIR = "./breeze-asr-25-final-corrected"

# 使用 train.py 的成功參數
QUICK_TEST_RATIO = 1  # 快速測試使用 10% 資料
QUICK_MAX_STEPS = 500  # 增加步數

# ==============================================================================
# 全域定義 - 基於 train.py 成功模式
# ==============================================================================


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """處理語音到序列資料的 Data Collator - 基於 train.py 成功版本"""

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def prepare_dataset_batched(batch, feature_extractor, tokenizer):
    """將一批音訊和文本資料即時轉換為模型輸入格式 - 基於 train.py 成功版本"""
    audio_list = batch["audio"]
    batch["input_features"] = feature_extractor(
        [x["array"] for x in audio_list], sampling_rate=audio_list[0]["sampling_rate"]
    ).input_features
    batch["labels"] = tokenizer(
        batch["中文意譯"], max_length=448, truncation=True
    ).input_ids
    return batch


def compute_metrics(pred, tokenizer):
    """計算 WER 指標 - 基於 train.py 成功版本"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    metric = evaluate.load("wer")
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ==============================================================================
# Colab 環境設定
# ==============================================================================


def setup_colab_environment():
    """設定 Colab 環境"""
    print("🔧 設定 Colab 環境...")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,max_split_size_mb:128"
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

    try:
        import sys

        if "google.colab" in sys.modules:
            from google.colab import drive

            drive.mount("/content/drive")
            print("✅ Google Drive 已掛載")
        else:
            print("ℹ️  非 Colab 環境，跳過 Drive 掛載")
    except ImportError:
        print("ℹ️  無法掛載 Google Drive (可能已掛載)")

    print("✅ Colab 環境設定完成")


# ==============================================================================
# 資料處理 - 基於 train.py 成功模式
# ==============================================================================


def load_and_clean_data():
    """載入並清理資料 - 基於 train.py 成功模式"""
    print("📊 載入資料...")

    if not Path(TRAIN_CSV).exists() or not Path(TEST_CSV).exists():
        raise FileNotFoundError(f"找不到 {TRAIN_CSV} 或 {TEST_CSV}")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    required_cols = ["file", "中文意譯"]
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"缺少必要欄位: {col}")

    # 移除空值
    train_df = train_df.dropna(subset=required_cols)
    test_df = test_df.dropna(subset=required_cols)

    # 限制文字長度
    train_df = train_df[train_df["中文意譯"].str.len() < 200]
    test_df = test_df[test_df["中文意譯"].str.len() < 200]

    # 快速測試採樣
    train_size = len(train_df)
    test_size = len(test_df)

    train_sample_size = max(1, int(train_size * QUICK_TEST_RATIO))
    test_sample_size = max(1, int(test_size * QUICK_TEST_RATIO))

    train_df = train_df.sample(n=train_sample_size, random_state=42)
    test_df = test_df.sample(n=test_sample_size, random_state=42)

    # 修正音訊檔案路徑 - 使用原本的位置
    print("🔧 修正音訊檔案路徑...")

    def fix_audio_path(path_str):
        if not isinstance(path_str, str):
            return path_str

        path_str = str(path_str).strip()

        # 如果已經是 Google Drive 路徑，直接返回
        if path_str.startswith("/content/drive/MyDrive/"):
            return path_str

        # 處理 Windows 路徑
        if path_str.startswith(("C:", "D:", "E:", "F:")):
            path_parts = Path(path_str).parts
            try:
                # 尋找 standard 目錄
                standard_idx = path_parts.index("standard")
                if standard_idx + 1 < len(path_parts):
                    relative_path = "/".join(path_parts[standard_idx:])
                    return f"/content/drive/MyDrive/{relative_path}"
            except ValueError:
                pass

            # 如果找不到 standard，使用檔案名
            filename = Path(path_str).name
            return f"/content/drive/MyDrive/{filename}"

        # 處理相對路徑
        if not path_str.startswith("/"):
            return f"/content/drive/MyDrive/{path_str}"

        return path_str

    train_df["file"] = train_df["file"].apply(fix_audio_path)
    test_df["file"] = test_df["file"].apply(fix_audio_path)

    sample_path = train_df["file"].iloc[0] if len(train_df) > 0 else ""
    print(f"   範例路徑: {sample_path}")

    print(f"✅ 原始訓練資料: {train_size} 筆")
    print(f"✅ 原始測試資料: {test_size} 筆")
    print(f"✅ 快速測試訓練資料: {len(train_df)} 筆 ({QUICK_TEST_RATIO*100}%)")
    print(f"✅ 快速測試測試資料: {len(test_df)} 筆 ({QUICK_TEST_RATIO*100}%)")

    return train_df, test_df


# ==============================================================================
# 主訓練函數 - 基於 train.py 成功模式
# ==============================================================================


def main():
    """主訓練流程 - 基於 train.py 成功模式"""
    print("🚀 Breeze-ASR-25 最終修正版本開始")
    print("=" * 60)
    print(f"🎯 使用資料比例: {QUICK_TEST_RATIO*100}%")
    print(f"🎯 最大訓練步數: {QUICK_MAX_STEPS}")

    setup_colab_environment()

    # 載入資料
    train_df, test_df = load_and_clean_data()

    print("🤖 載入模型...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        # torch_dtype=torch.float16,  # 使用 FP16 載入
        low_cpu_mem_usage=True,  # 低 CPU 記憶體使用
        use_cache=False,  # 關閉快取
    )

    # 設定模型配置 - 基於 train.py 成功模式
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    print("📋 創建資料集...")

    # 創建資料集 - 基於 train.py 成功模式
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 轉換為音頻格式
    train_dataset = train_dataset.cast_column("file", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("file", Audio(sampling_rate=16000))

    # 重命名欄位
    train_dataset = train_dataset.rename_column("file", "audio")
    test_dataset = test_dataset.rename_column("file", "audio")

    # 使用 .with_transform() 確保記憶體穩定 - 基於 train.py 成功模式
    prepare_fn = partial(
        prepare_dataset_batched,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
    )

    train_dataset = train_dataset.with_transform(prepare_fn)
    test_dataset = test_dataset.with_transform(prepare_fn)

    print("✅ 即時轉換已設定")

    # 建立訓練元件 - 基於 train.py 成功模式
    print("🔧 建立訓練元件...")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics_fn = partial(compute_metrics, tokenizer=processor.tokenizer)

    # 訓練參數 - 基於 train.py 成功模式但調整為 Breeze 適用
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # 批次大小設定 - 基於 train.py 成功模式
        per_device_train_batch_size=2,  # 從 4 降到 1
        per_device_eval_batch_size=2,  # 從 4 降到 1
        gradient_accumulation_steps=8,  # 從 4 增加到 8 (保持有效批次大小)
        # 學習率和步數設定
        learning_rate=1e-5,
        warmup_steps=100,
        max_steps=QUICK_MAX_STEPS,
        # 評估設定
        eval_strategy="steps",
        eval_steps=100,
        predict_with_generate=True,
        generation_max_length=64,  # 減少生成長度，降低記憶體使用
        # 保存設定
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        # 優化設定 - 基於 train.py 成功模式
        gradient_checkpointing=False,  # 啟用梯度檢查點，用計算時間換取記憶體
        fp16=False,  # 啟用 FP16 可以減少約 50% 記憶體使用
        dataloader_num_workers=0,  # 關閉多進程載入，減少記憶體使用
        # 其他設定
        logging_steps=25,
        report_to=[],
        remove_unused_columns=False,
        push_to_hub=True,
    )

    # 創建訓練器 - 基於 train.py 成功模式
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=processor.feature_extractor,
    )

    # 顯示訓練資訊
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    effective_batch = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
    )

    print(f"\n📈 最終修正版本資訊:")
    print(f"   模型參數: {total_params:,}")
    print(f"   可訓練參數: {trainable_params:,}")
    print(f"   有效批次: {effective_batch}")
    print(f"   最大步數: {QUICK_MAX_STEPS}")
    print(f"   學習率: {training_args.learning_rate}")
    print(f"   評估頻率: 每 {training_args.eval_steps} 步")
    print(f"   使用 train.py 成功模式")

    print("\n🚀 開始最終修正版本訓練...")
    try:
        trainer.train()
        print("✅ 最終修正版本訓練完成")

        # 保存模型
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)

        # 最終評估
        metrics = trainer.evaluate()
        print(f"🎯 最終 WER: {metrics.get('eval_wer', 'N/A'):.2f}%")

        print(f"\n💾 最終修正版本模型已保存至: {OUTPUT_DIR}")

    except Exception as e:
        print(f"❌ 訓練錯誤: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
