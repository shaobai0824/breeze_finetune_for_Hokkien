# ==============================================================================
# 檔案：evaluate_test_dataset.py
# 描述：專門用於評估獨立測試資料集的語義相似度評估器
# ==============================================================================
import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Audio
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class TestDatasetEvaluator:
    """測試資料集語義相似度評估器"""

    def __init__(self, embedding_model_name="shibing624/text2vec-base-chinese"):
        """
        初始化評估器

        Args:
            embedding_model_name (str): 向量模型名稱
        """
        print(f"正在載入向量模型：{embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name

    def compute_similarity(self, text1, text2):
        """計算兩個文本的語義相似度"""
        if not text1.strip() or not text2.strip():
            return 0.0

        embeddings = self.embedding_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)


def is_huggingface_model(model_path):
    """檢查是否為 Hugging Face Hub 上的模型"""
    # 如果路徑包含 "/" 且不是絕對路徑，很可能是 HF Hub 模型
    return (
        "/" in model_path
        and not os.path.isabs(model_path)
        and not os.path.exists(model_path)
    )


def load_test_dataset(csv_path, audio_folder):
    """
    載入測試資料集

    Args:
        csv_path (str): CSV檔案路徑
        audio_folder (str): 音訊檔案資料夾路徑

    Returns:
        dict: 測試資料字典
    """
    print(f"載入測試資料集從：{csv_path}")

    # 讀取CSV檔案
    df = pd.read_csv(csv_path)

    # 檢查必要欄位
    required_columns = ["filename", "transcription"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV檔案缺少必要欄位：{missing_columns}")

    # 支援的音訊格式
    supported_formats = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]

    # 檢查音訊檔案是否存在
    existing_files = []
    missing_files = []

    for _, row in df.iterrows():
        filename = row["filename"]
        audio_path = os.path.join(audio_folder, filename)

        if os.path.exists(audio_path):
            # 檢查檔案格式
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_formats:
                existing_files.append(row)
            else:
                print(f"⚠️  警告：不支援的音訊格式 {filename} ({file_ext})")
                missing_files.append(filename)
        else:
            missing_files.append(filename)

    if missing_files:
        print(f"⚠️  警告：找不到或不支援 {len(missing_files)} 個音訊檔案：")
        for file in missing_files[:5]:  # 只顯示前5個
            print(f"   - {file}")
        if len(missing_files) > 5:
            print(f"   ... 還有 {len(missing_files) - 5} 個檔案")

    print(f"✅ 成功載入 {len(existing_files)} 個有效測試樣本")
    print(f"📁 支援的音訊格式: {', '.join(supported_formats)}")

    # 建立資料字典
    test_data = {
        "files": existing_files,
        "total_files": len(existing_files),
        "missing_files": missing_files,
        "has_domain": "domain" in df.columns,
        "has_speaker": "speaker_id" in df.columns,
        "has_duration": "duration" in df.columns,
    }

    return test_data


def evaluate_test_dataset(
    model_path: str,
    test_csv_path: str,
    test_audio_folder: str,
    embedding_model: str = "shibing624/text2vec-base-chinese",
    output_dir: str = "evaluation_results",
    batch_size: int = 1,
    language: str = "zh",
):
    """
    評估測試資料集

    Args:
        model_path (str): 語音辨識模型路徑
        test_csv_path (str): 測試資料CSV檔案路徑
        test_audio_folder (str): 測試音訊資料夾路徑
        embedding_model (str): 向量模型名稱
        output_dir (str): 結果輸出資料夾
        batch_size (int): 批次大小
        language (str): 語言設定
    """

    # --- 1. 環境初始化 ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  使用設備：{device}")
    print(f"🎯 模型路徑：{model_path}")
    print(f"📊 測試資料：{test_csv_path}")
    print(f"🎵 音訊資料夾：{test_audio_folder}")
    print()

    # --- 2. 載入模型 ---
    print("🤖 載入語音辨識模型...")

    # 檢查模型類型
    if is_huggingface_model(model_path):
        print(f"🌐 檢測到 Hugging Face Hub 模型：{model_path}")
        print("📥 正在從網路下載模型...")
    else:
        print(f"💻 檢測到本地模型：{model_path}")

    try:
        processor = WhisperProcessor.from_pretrained(model_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
        print("✅ 語音辨識模型載入成功")

        # 顯示模型資訊
        if hasattr(model.config, "name_or_path"):
            print(f"📋 模型名稱：{model.config.name_or_path}")
        if hasattr(model.config, "_name_or_path"):
            print(f"📋 模型路徑：{model.config._name_or_path}")

    except Exception as e:
        print(f"❌ 語音辨識模型載入失敗：{e}")
        if is_huggingface_model(model_path):
            print("💡 提示：")
            print("   - 請檢查網路連接")
            print("   - 確認模型名稱是否正確")
            print("   - 某些模型可能需要登入 Hugging Face")
        else:
            print("💡 提示：")
            print("   - 檢查模型路徑是否正確")
            print("   - 確認模型檔案是否完整")
        return None

    print("\n🧠 載入語義相似度評估器...")
    try:
        similarity_evaluator = TestDatasetEvaluator(embedding_model)
        print("✅ 語義相似度評估器載入成功")
    except Exception as e:
        print(f"❌ 語義相似度評估器載入失敗：{e}")
        return None

    # --- 3. 載入測試資料 ---
    print("\n📂 載入測試資料集...")
    try:
        test_data = load_test_dataset(test_csv_path, test_audio_folder)
    except Exception as e:
        print(f"❌ 測試資料載入失敗：{e}")
        return None

    if test_data["total_files"] == 0:
        print("❌ 沒有可用的測試檔案")
        return None

    # --- 4. 進行評估 ---
    print(f"\n🔍 開始評估 {test_data['total_files']} 個測試樣本...")

    predictions = []
    references = []
    similarity_scores = []
    detailed_results = []
    domain_results = defaultdict(list) if test_data["has_domain"] else None
    speaker_results = defaultdict(list) if test_data["has_speaker"] else None

    audio_loader = Audio(sampling_rate=16000)

    # 評估進度條
    for file_data in tqdm(test_data["files"], desc="評估進度"):
        filename = file_data["filename"]
        reference_text = file_data["transcription"]
        audio_path = os.path.join(test_audio_folder, filename)

        try:
            # 載入音訊（支援多種格式）
            try:
                audio_input = audio_loader.decode_example(
                    audio_loader.encode_example(audio_path)
                )
            except Exception as audio_error:
                print(f"\n⚠️  無法載入音訊檔案 {filename}: {audio_error}")
                continue

            # 檢查音訊品質
            audio_array = audio_input["array"]
            if len(audio_array) == 0:
                print(f"\n⚠️  音訊檔案 {filename} 為空")
                continue

            # 處理音訊特徵
            input_features = processor(
                audio_array,
                sampling_rate=audio_input["sampling_rate"],
                return_tensors="pt",
            ).input_features.to(device)

            # 語音辨識
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    max_length=448,  # 限制最大長度
                    do_sample=False,  # 使用貪婪解碼
                    num_beams=1,  # 簡化解碼過程
                )
                predicted_text = processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0].strip()

            # 檢查預測結果
            if not predicted_text:
                predicted_text = "[無法識別]"
                print(f"\n⚠️  檔案 {filename} 無法產生識別結果")

            # 計算語義相似度
            similarity_score = similarity_evaluator.compute_similarity(
                reference_text, predicted_text
            )

            # 收集結果
            predictions.append(predicted_text)
            references.append(reference_text)
            similarity_scores.append(similarity_score)

            # 詳細結果
            result_item = {
                "filename": filename,
                "reference": reference_text,
                "prediction": predicted_text,
                "similarity_score": similarity_score,
                "audio_duration": len(audio_array) / audio_input["sampling_rate"],
            }

            # 添加額外資訊
            if test_data["has_domain"]:
                domain = file_data.get("domain", "unknown")
                result_item["domain"] = domain
                domain_results[domain].append(similarity_score)

            if test_data["has_speaker"]:
                speaker = file_data.get("speaker_id", "unknown")
                result_item["speaker_id"] = speaker
                speaker_results[speaker].append(similarity_score)

            if test_data["has_duration"]:
                result_item["expected_duration"] = file_data.get("duration", 0)

            detailed_results.append(result_item)

        except Exception as e:
            print(f"\n⚠️  處理檔案 {filename} 時發生錯誤: {e}")
            # 添加失敗記錄
            detailed_results.append(
                {
                    "filename": filename,
                    "reference": reference_text,
                    "prediction": f"[處理失敗: {str(e)}]",
                    "similarity_score": 0.0,
                    "error": str(e),
                }
            )
            continue

    # --- 5. 計算評估指標 ---
    if not predictions:
        print("❌ 沒有成功處理的檔案")
        return None

    print(f"\n✅ 成功處理 {len(predictions)} 個檔案")
    print("\n📊 計算評估指標...")

    # WER 計算
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references) * 100

    # 語義相似度統計
    mean_similarity = np.mean(similarity_scores)
    std_similarity = np.std(similarity_scores)
    median_similarity = np.median(similarity_scores)

    # 分類統計
    high_similarity = [s for s in similarity_scores if s >= 0.8]
    moderate_similarity = [s for s in similarity_scores if 0.6 <= s < 0.8]
    low_similarity = [s for s in similarity_scores if s < 0.6]

    # --- 6. 顯示結果 ---
    print("\n" + "=" * 80)
    print("                        測試資料集評估結果")
    print("=" * 80)
    print(f"📁 測試資料集：{os.path.basename(test_csv_path)}")
    print(f"🤖 語音模型：{os.path.basename(model_path)}")
    print(f"🧠 向量模型：{embedding_model}")
    print(f"📊 評估樣本數：{len(predictions)}")
    print()

    print("🎯 傳統指標:")
    print(f"   字詞錯誤率 (WER): {wer:.2f}%")
    print()

    print("🧠 語義相似度指標:")
    print(f"   平均相似度: {mean_similarity:.4f}")
    print(f"   中位數相似度: {median_similarity:.4f}")
    print(f"   標準差: {std_similarity:.4f}")
    print()

    print("📈 相似度分布:")
    print(
        f"   高相似度 (≥0.8): {len(high_similarity)} 個 ({len(high_similarity)/len(predictions)*100:.1f}%)"
    )
    print(
        f"   中等相似度 (0.6-0.8): {len(moderate_similarity)} 個 ({len(moderate_similarity)/len(predictions)*100:.1f}%)"
    )
    print(
        f"   低相似度 (<0.6): {len(low_similarity)} 個 ({len(low_similarity)/len(predictions)*100:.1f}%)"
    )

    # 領域分析
    if domain_results:
        print("\n📋 按領域分析:")
        for domain, scores in domain_results.items():
            avg_score = np.mean(scores)
            print(f"   {domain}: {avg_score:.3f} ({len(scores)} 樣本)")

    # 說話者分析
    if speaker_results:
        print("\n🎤 按說話者分析:")
        for speaker, scores in speaker_results.items():
            avg_score = np.mean(scores)
            print(f"   {speaker}: {avg_score:.3f} ({len(scores)} 樣本)")

    print("=" * 80)

    # --- 7. 儲存結果 ---
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 儲存詳細結果
    results_file = os.path.join(output_dir, f"test_evaluation_{timestamp}.json")

    results_summary = {
        "evaluation_info": {
            "timestamp": timestamp,
            "model_path": model_path,
            "embedding_model": embedding_model,
            "test_csv": test_csv_path,
            "test_audio_folder": test_audio_folder,
            "total_samples": len(predictions),
        },
        "metrics": {
            "wer": wer,
            "mean_similarity": mean_similarity,
            "median_similarity": median_similarity,
            "std_similarity": std_similarity,
            "high_similarity_count": len(high_similarity),
            "moderate_similarity_count": len(moderate_similarity),
            "low_similarity_count": len(low_similarity),
        },
        "domain_analysis": (
            {domain: np.mean(scores) for domain, scores in domain_results.items()}
            if domain_results
            else {}
        ),
        "speaker_analysis": (
            {speaker: np.mean(scores) for speaker, scores in speaker_results.items()}
            if speaker_results
            else {}
        ),
        "detailed_results": detailed_results,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    print(f"\n💾 詳細結果已儲存至：{results_file}")

    # 儲存CSV格式結果
    csv_file = os.path.join(output_dir, f"test_evaluation_{timestamp}.csv")
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(csv_file, index=False, encoding="utf-8")
    print(f"📊 CSV結果已儲存至：{csv_file}")

    # 顯示最佳和最差範例
    sorted_results = sorted(
        detailed_results, key=lambda x: x["similarity_score"], reverse=True
    )

    print(f"\n🏆 相似度最高的 3 個範例:")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"  {i}. [{result['filename']}] 相似度: {result['similarity_score']:.4f}")
        print(f"     參考: {result['reference']}")
        print(f"     預測: {result['prediction']}")
        print()

    print(f"⚠️  相似度最低的 3 個範例:")
    for i, result in enumerate(sorted_results[-3:], 1):
        print(f"  {i}. [{result['filename']}] 相似度: {result['similarity_score']:.4f}")
        print(f"     參考: {result['reference']}")
        print(f"     預測: {result['prediction']}")
        print()

    return results_summary


def main():
    """主函數 - 支援命令列參數"""
    parser = argparse.ArgumentParser(description="測試資料集語義相似度評估")
    parser.add_argument("--model_path", required=True, help="語音辨識模型路徑")
    parser.add_argument("--test_csv", required=True, help="測試資料CSV檔案路徑")
    parser.add_argument("--test_audio", required=True, help="測試音訊資料夾路徑")
    parser.add_argument(
        "--embedding_model",
        default="shibing624/text2vec-base-chinese",
        help="向量模型名稱",
    )
    parser.add_argument(
        "--output_dir", default="evaluation_results", help="結果輸出資料夾"
    )
    parser.add_argument("--language", default="zh", help="語言設定")

    args = parser.parse_args()

    # 執行評估
    results = evaluate_test_dataset(
        model_path=args.model_path,
        test_csv_path=args.test_csv,
        test_audio_folder=args.test_audio,
        embedding_model=args.embedding_model,
        output_dir=args.output_dir,
        language=args.language,
    )

    if results:
        print("\n🎉 評估完成！")
    else:
        print("\n❌ 評估失敗")


if __name__ == "__main__":
    # 可以直接設定參數或使用命令列
    import sys

    if len(sys.argv) == 1:
        # 直接執行模式 - 在這裡設定你的參數
        print("直接執行模式 - 請修改以下參數:")
        print()

        # === 請修改以下參數 ===
        MODEL_PATH = "shaobai880824/breeze-asr-25-chinese-full"
        TEST_CSV = "test_references_wav.csv"
        TEST_AUDIO_FOLDER = "debug_audio_wav"
        EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
        OUTPUT_DIR = "evaluation_results"

        print(f"模型路徑: {MODEL_PATH}")
        print(f"測試CSV: {TEST_CSV}")
        print(f"音訊資料夾: {TEST_AUDIO_FOLDER}")
        print(f"向量模型: {EMBEDDING_MODEL}")
        print(f"輸出資料夾: {OUTPUT_DIR}")
        print()

        # 檢查模型是否可用
        if not is_huggingface_model(MODEL_PATH) and not os.path.exists(MODEL_PATH):
            print(f"❌ 找不到本地模型路徑: {MODEL_PATH}")
            exit(1)
        elif is_huggingface_model(MODEL_PATH):
            print(f"🌐 將使用 Hugging Face Hub 模型: {MODEL_PATH}")
        if not os.path.exists(TEST_CSV):
            print(f"❌ 找不到測試CSV: {TEST_CSV}")
            print("請參考 test_references_example.csv 建立你的測試資料")
            exit(1)
        if not os.path.exists(TEST_AUDIO_FOLDER):
            print(f"❌ 找不到音訊資料夾: {TEST_AUDIO_FOLDER}")
            exit(1)

        results = evaluate_test_dataset(
            model_path=MODEL_PATH,
            test_csv_path=TEST_CSV,
            test_audio_folder=TEST_AUDIO_FOLDER,
            embedding_model=EMBEDDING_MODEL,
            output_dir=OUTPUT_DIR,
        )
    else:
        # 命令列模式
        main()
