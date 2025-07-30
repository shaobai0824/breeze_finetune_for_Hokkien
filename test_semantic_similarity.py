# ==============================================================================
# 檔案：test_semantic_similarity.py
# 描述：測試語義相似度評估功能，確保所有套件安裝正確且功能正常
# ==============================================================================

import time

from evaluate_with_semantic_similarity import SemanticSimilarityEvaluator


def test_similarity_function():
    """測試語義相似度計算功能"""
    print("=== 語義相似度評估功能測試 ===\n")

    # 測試用例 - 中文語音辨識場景
    test_cases = [
        # (參考文本, 預測文本, 預期相似度範圍描述)
        ("今天天氣很好", "今天天氣很好", "完全相同 (期望接近1.0)"),
        ("今天天氣很好", "今日天氣很棒", "語義相似 (期望>0.7)"),
        ("今天天氣很好", "今天氣溫非常高", "部分相關 (期望0.4-0.7)"),
        ("今天天氣很好", "明天會下雨", "語義不同 (期望<0.5)"),
        ("我要買一杯咖啡", "我想要購買咖啡", "語義相同 (期望>0.8)"),
        ("請問現在幾點", "請告訴我現在的時間", "語義相同 (期望>0.8)"),
        ("你好嗎", "再見", "語義無關 (期望<0.3)"),
        ("一二三四五", "12345", "數字表達 (期望中等)"),
    ]

    try:
        # 初始化評估器
        print("正在載入語義相似度評估器...")
        evaluator = SemanticSimilarityEvaluator("shibing624/text2vec-base-chinese")
        print("✅ 評估器載入成功！\n")

        print("開始測試相似度計算...")
        print("-" * 80)

        for i, (ref, pred, description) in enumerate(test_cases, 1):
            start_time = time.time()
            similarity = evaluator.compute_similarity(ref, pred)
            end_time = time.time()

            print(f"測試 {i}: {description}")
            print(f"  參考文本: 「{ref}」")
            print(f"  預測文本: 「{pred}」")
            print(f"  相似度分數: {similarity:.4f}")
            print(f"  計算時間: {(end_time - start_time)*1000:.1f}ms")
            print()

        print("-" * 80)
        print("🎉 所有測試完成！")

        # 效能測試
        print("\n=== 效能測試 ===")
        test_ref = "這是一個效能測試的範例句子"
        test_pred = "這是效能測試的示例語句"

        start_time = time.time()
        for _ in range(10):
            evaluator.compute_similarity(test_ref, test_pred)
        end_time = time.time()

        avg_time = (end_time - start_time) / 10 * 1000
        print(f"平均計算時間 (10次): {avg_time:.1f}ms")

        return True

    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        print("請確保已安裝所有必要套件：")
        print("pip install sentence-transformers scikit-learn scipy numpy")
        return False

    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False


def test_different_embedding_models():
    """測試不同的向量模型"""
    print("\n=== 測試不同向量模型 ===\n")

    # 推薦的中文向量模型
    models_to_test = [
        "shibing624/text2vec-base-chinese",
        # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 多語言模型
    ]

    test_ref = "我今天很開心"
    test_pred = "我今日非常高興"

    for model_name in models_to_test:
        try:
            print(f"測試模型: {model_name}")
            evaluator = SemanticSimilarityEvaluator(model_name)

            start_time = time.time()
            similarity = evaluator.compute_similarity(test_ref, test_pred)
            end_time = time.time()

            print(f"  相似度: {similarity:.4f}")
            print(f"  載入+計算時間: {(end_time - start_time)*1000:.1f}ms")
            print("  ✅ 測試成功\n")

        except Exception as e:
            print(f"  ❌ 測試失敗: {e}\n")


if __name__ == "__main__":
    print("開始語義相似度評估系統測試...\n")

    # 基礎功能測試
    success = test_similarity_function()

    if success:
        # 不同模型測試
        test_different_embedding_models()

        print("=" * 60)
        print("🎊 恭喜！語義相似度評估系統已準備就緒")
        print(
            "您現在可以使用 evaluate_with_semantic_similarity.py 來評估您的語音辨識模型"
        )
        print("=" * 60)
    else:
        print("❌ 測試失敗，請檢查套件安裝")
