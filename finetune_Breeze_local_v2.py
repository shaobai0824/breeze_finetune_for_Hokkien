#!/usr/bin/env python3
"""
Breeze-ASR-25 本機訓練版本 - 基於 train.py 成功模式
專門用於 /standard 目錄的檔案位置
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

from evaluate_with_semantic_similarity import (
    SemanticSimilarityEvaluator,
)  # 導入語義相似度評估器

# ==============================================================================
# 基本配置 - 本機版本
# ==============================================================================

MODEL_ID = "shaobai880824/breeze-asr-25-local-hokkien_v1"
DATA_CSV = (
    "audio_1_converted_merged_zh.csv"  # <--- 修改這裡，指定您包含所有資料的 CSV 檔案
)
TEST_SPLIT_RATIO = 0.2  # 測試集佔總資料的比例 (例如 0.2 表示 20%)
OUTPUT_DIR = "./breeze-asr-25-local-hokkien_v2"

# 本機訓練參數
QUICK_TEST_RATIO = 1  # 快速測試使用 10% 資料
QUICK_MAX_STEPS = 500  # 訓練步數

# ==============================================================================
# 全域定義 - 基於 train.py 成功模式
# ==============================================================================

# 為了效率，在此處初始化評估器，使其在整個訓練過程中僅載入一次
print("⏳ 正在預先載入語義相似度評估模型...")
semantic_evaluator = SemanticSimilarityEvaluator("shibing624/text2vec-base-chinese")
print("✅ 語義相似度評估模型載入完成。")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """處理語音到序列資料的 Data Collator - 基於 train.py 成功版本"""

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        # 1. 處理 input_features (音訊聲學特徵)
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        # 2. 處理 labels (文本標籤)
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 3. 處理標籤中的 -100
        # 將填充部分（attention_mask 為 0）的標籤替換為 -100。
        # 在 PyTorch 的交叉熵損失函數中，-100 是一個特殊的 ignore_index，
        # 意味著這些位置的損失不會被計算，這對於填充部分非常重要。
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # 4. 處理開頭的 BOS token (如果存在)
        # 檢查所有序列的第一個 token 是否都是 BOS token。
        # 這是針對某些模型（如 Whisper）的訓練習慣，模型在訓練時通常預期標籤不包含開頭的 BOS token。
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def prepare_dataset_batched(batch, feature_extractor, tokenizer):
    """將一批音訊和文本資料即時轉換為模型輸入格式 - 基於 train.py 成功版本"""
    audio_list = batch["audio"]
    batch["input_features"] = feature_extractor(
        # ## 修改點: 這裡不再需要擔心 audio_list[0] 是 None，因為壞資料已被過濾
        [x["array"] for x in audio_list],
        sampling_rate=audio_list[0]["sampling_rate"],
    ).input_features
    batch["labels"] = tokenizer(
        batch["漢字"], max_length=448, truncation=True
    ).input_ids
    return batch


def compute_metrics(pred, tokenizer):
    """計算語義相似度指標"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 將 -100 替換為 pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # 解碼預測和標籤
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 使用全域的 semantic_evaluator 計算相似度
    similarities = [
        semantic_evaluator.compute_similarity(ref, pred)
        for ref, pred in zip(label_str, pred_str)
    ]

    # 計算平均相似度
    avg_similarity = np.mean(similarities) if similarities else 0.0

    return {"semantic_similarity": avg_similarity}


# ==============================================================================
# 本機環境設定
# ==============================================================================


def setup_local_environment():
    """設定本機環境"""
    print("🔧 設定本機環境...")

    # 設定 CUDA 記憶體分配策略
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,max_split_size_mb:128"
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print(f"✅ GPU 可用: {torch.cuda.get_device_name(0)}")
        print(
            f"✅ GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        print("⚠️  GPU 不可用，將使用 CPU 訓練")

    print("✅ 本機環境設定完成")


# ==============================================================================
# 資料處理 - 本機版本
# ==============================================================================


def load_and_clean_data():
    """載入、清理並從單一檔案拆分資料 - 本機版本"""
    print("📊 載入、清理並拆分資料...")

    if not Path(DATA_CSV).exists():
        raise FileNotFoundError(f"找不到指定的資料檔案: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)

    required_cols = ["檔案位置", "漢字"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要欄位: {col}")

    # 在拆分前對整個資料集進行清理
    df.dropna(subset=required_cols, inplace=True)
    df = df[df["漢字"].str.len() < 200].copy()

    print(f"✅ 清理後總資料: {len(df)} 筆")

    # 根據比例拆分訓練集和測試集
    test_df = df.sample(frac=TEST_SPLIT_RATIO, random_state=42)
    train_df = df.drop(test_df.index)

    train_size_after_split = len(train_df)
    test_size_after_split = len(test_df)

    print(f"\n✅ 拆分後訓練資料: {train_size_after_split} 筆")
    print(
        f"✅ 拆分後測試資料: {test_size_after_split} 筆 (目標比例: {TEST_SPLIT_RATIO*100:.1f}%)"
    )

    # 基於拆分後的資料，再進行快速測試的採樣
    if QUICK_TEST_RATIO < 1.0:
        train_df = train_df.sample(frac=QUICK_TEST_RATIO, random_state=42)
        test_df = test_df.sample(frac=QUICK_TEST_RATIO, random_state=42)
        print(f"✅ 快速測試模式已啟用 (使用 {QUICK_TEST_RATIO*100}%)")
        print(f"  - 使用訓練資料: {len(train_df)} 筆")
        print(f"  - 使用測試資料: {len(test_df)} 筆")

    # 修正音訊檔案路徑 - 本機 /standard 目錄
    print("🔧 修正音訊檔案路徑...")

    def fix_audio_path(path_str):
        if not isinstance(path_str, str):
            return path_str

        path_str = str(path_str).strip()

        # 如果已經是正確路徑，直接返回
        if Path(path_str).exists():
            return path_str

        # 處理 Windows 路徑
        if path_str.startswith(("C:", "D:", "E:", "F:")):
            path_parts = Path(path_str).parts
            try:
                # 尋找 standard 目錄
                standard_idx = path_parts.index("standard")
                if standard_idx + 1 < len(path_parts):
                    relative_path = "/".join(path_parts[standard_idx:])
                    # 使用本機 /standard 目錄
                    local_path = f"/standard/{relative_path}"
                    if Path(local_path).exists():
                        return local_path
            except ValueError:
                pass

            # 如果找不到 standard，使用檔案名
            filename = Path(path_str).name
            local_path = f"/standard/{filename}"
            if Path(local_path).exists():
                return local_path

        # 處理相對路徑
        if not path_str.startswith("/"):
            local_path = f"/standard/{path_str}"
            if Path(local_path).exists():
                return local_path

        # 如果都不存在，返回原始路徑
        return path_str

    train_df.loc[:, "檔案位置"] = train_df["檔案位置"].apply(fix_audio_path)
    test_df.loc[:, "檔案位置"] = test_df["檔案位置"].apply(fix_audio_path)

    sample_path = train_df["檔案位置"].iloc[0] if len(train_df) > 0 else ""
    print(f"  範例路徑: {sample_path}")

    return train_df, test_df


# ==============================================================================
# 主訓練函數 - 基於 train.py 成功模式
# ==============================================================================


def main():
    """主訓練流程 - 本機版本"""
    print("🚀 Breeze-ASR-25 本機訓練版本開始")
    print("=" * 60)
    print(f"🎯 使用資料比例: {QUICK_TEST_RATIO*100}%")
    print(f"🎯 最大訓練步數: {QUICK_MAX_STEPS}")

    setup_local_environment()

    # 載入資料
    train_df, test_df = load_and_clean_data()

    print("🤖 載入模型...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    # 設定模型配置 - 基於 train.py 成功模式
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    print("📋 創建資料集...")

    # 創建資料集 - 基於 train.py 成功模式
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # ## ==================== 修改點 START ==================== ##
    # ## 修正欄位名稱並加入資料過濾步驟來根除 'NoneType' 錯誤

    print("🔊 轉換音訊格式並驗證資料...")
    # 1. 轉換為音頻格式 (使用正確的欄位名 "檔案位置")
    #    這一步會將無法讀取的音檔路徑轉換為 None
    train_dataset = train_dataset.cast_column("檔案位置", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("檔案位置", Audio(sampling_rate=16000))

    # 2. 重命名欄位以符合後續處理
    train_dataset = train_dataset.rename_column("檔案位置", "audio")
    test_dataset = test_dataset.rename_column("檔案位置", "audio")

    # 3. 過濾掉值為 None 的無效音訊資料 (關鍵步驟)
    def is_audio_valid(example):
        return example["audio"] is not None

    original_train_len = len(train_dataset)
    original_test_len = len(test_dataset)

    train_dataset = train_dataset.filter(is_audio_valid, num_proc=4)  # 可調整 num_proc
    test_dataset = test_dataset.filter(is_audio_valid, num_proc=4)

    filtered_train_len = len(train_dataset)
    filtered_test_len = len(test_dataset)

    if original_train_len != filtered_train_len:
        print(
            f"🧹 從訓練集中移除了 {original_train_len - filtered_train_len} 筆無效音訊。"
        )
    if original_test_len != filtered_test_len:
        print(
            f"🧹 從測試集中移除了 {original_test_len - filtered_test_len} 筆無效音訊。"
        )

    print(f"✅ 有效訓練資料: {filtered_train_len} 筆")
    print(f"✅ 有效測試資料: {filtered_test_len} 筆")

    # ## ===================== 修改點 END ===================== ##

    # 使用 .with_transform() 確保記憶體穩定 - 基於 train.py 成功模式
    prepare_fn = partial(
        prepare_dataset_batched,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
    )

    # ## 修改點: 使用已過濾的 dataset
    train_dataset = train_dataset.with_transform(prepare_fn)
    test_dataset = test_dataset.with_transform(prepare_fn)

    print("✅ 即時轉換已設定")

    # 建立訓練元件 - 基於 train.py 成功模式
    print("🔧 建立訓練元件...")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics_fn = partial(compute_metrics, tokenizer=processor.tokenizer)

    # 訓練參數 - 基於 train.py 成功模式，適合本機環境
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # 批次大小設定 - 基於 train.py 成功模式
        per_device_train_batch_size=1,  # 使用 train.py 的設定
        per_device_eval_batch_size=1,  # 使用 train.py 的設定
        gradient_accumulation_steps=16,  # 使用 train.py 的設定
        # 學習率和步數設定
        learning_rate=1e-5,
        warmup_steps=100,
        max_steps=QUICK_MAX_STEPS,
        # 評估設定
        eval_strategy="steps",
        eval_steps=100,
        predict_with_generate=True,
        generation_max_length=64,  # 使用 train.py 的設定
        # 保存設定
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="semantic_similarity",  # <-- 改為語義相似度
        greater_is_better=True,  # <-- 分數越高越好
        # 優化設定 - 基於 train.py 成功模式
        gradient_checkpointing=False,
        fp16=False,  # 使用 train.py 的設定
        dataloader_num_workers=0,  # 使用 train.py 的設定
        # 其他設定
        logging_steps=10,
        report_to=[],
        remove_unused_columns=False,
        push_to_hub=True,  # ## 修改點：本機版本預設不上傳到Hugging Face Hub
    )

    # 創建訓練器 - 基於 train.py 成功模式
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,  # ## 修改點: 使用已過濾的 dataset
        eval_dataset=test_dataset,  # ## 修改點: 使用已過濾的 dataset
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=processor.tokenizer,  # <-- 修正：應傳遞 tokenizer
    )

    # 顯示訓練資訊
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    effective_batch = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
    )

    print(f"\n📈 本機訓練版本資訊:")
    print(f"  模型參數: {total_params:,}")
    print(f"  可訓練參數: {trainable_params:,}")
    print(f"  有效批次: {effective_batch}")
    print(f"  最大步數: {QUICK_MAX_STEPS}")
    print(f"  學習率: {training_args.learning_rate}")
    print(f"  評估頻率: 每 {training_args.eval_steps} 步")
    print(f"  使用 train.py 成功模式")
    print(f"  檔案位置: /standard 目錄")

    print("\n🚀 開始本機訓練...")
    try:
        trainer.train()
        print("✅ 本機訓練完成")

        # 保存模型
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)

        # 最終評估
        metrics = trainer.evaluate()
        final_metric_name = f"eval_{training_args.metric_for_best_model}"
        print(
            f"🎯 最終 {training_args.metric_for_best_model}: {metrics.get(final_metric_name, 'N/A'):.4f}"
        )

        print(f"\n💾 本機訓練模型已保存至: {OUTPUT_DIR}")

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
