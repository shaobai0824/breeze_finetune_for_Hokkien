# Breeze-ASR-25 本地微調與評估工具

本專案提供一套完整的工具鏈，用於在本地環境對 MediaTek-Research 的 `Breeze-ASR-25` 語音辨識模型進行微調，並使用**語義相似度**等進階指標對其性能進行全面評估。

## 核心功能

- **🔥 本地微調**: 使用 `finetune_Breeze_local.py` 在您自己的資料集上對 Breeze 模型進行微調。
- **📊 進階評估**: 使用 `evaluate_test_dataset.py` 評估模型，不僅計算傳統的字詞錯誤率 (WER)，還引入**語義相似度**，更準確地衡量模型理解能力。
- **📁 詳盡報告**: 產生 JSON 和 CSV 格式的詳細評估報告，包含每個檔案的辨識結果、相似度分數，並提供領域 (Domain) 和說話者 (Speaker) 的分類統計。
- **⚙️ 操作簡易**: 腳本參數化，易於配置與執行。

## 工作流程

```
[準備訓練資料] -> [執行微調腳本] -> [產生自訂模型] -> [準備測試資料] -> [執行評估腳本] -> [獲取分析報告]
      |                 |                   |                 |                 |                   |
  (音訊檔 & CSV) (finetune_Breeze_local.py)  (模型資料夾)   (音訊檔 & CSV) (evaluate_test_dataset.py)  (evaluation_results)
```

## 環境設定

1.  **克隆專案**
    ```bash
    git clone <repository_url>
    cd audio_model
    ```

2.  **安裝依賴**
    建議在虛擬環境中安裝。
    ```bash
    pip install -r requirements.txt
    ```
    *注意：請確保 `requirements.txt` 檔案包含所有必要套件，如 `torch`, `transformers`, `datasets`, `evaluate`, `sentence-transformers`, `pandas` 等。*

---

## 第一部分：模型微調 (`finetune_Breeze_local.py`)

此腳本專為在本地機器（特別是具有 NVIDIA GPU 的環境）上微調 `Breeze-ASR-25` 模型而設計。

### 1. 資料準備

- **音訊檔案**: 將所有訓練和測試用的 `.wav` 音訊檔案放置在專案根目錄下的 `standard/` 資料夾中。腳本會自動在此目錄下查找檔案。
- **中繼資料 (CSV)**: 在專案根目錄下準備兩個 CSV 檔案：
    - `metadata_train_fixed.csv` (訓練集)
    - `metadata_test_fixed.csv` (測試集)

    CSV 檔案必須包含以下兩欄：
    - `file`: 音訊檔案的路徑 (相對於 `standard/` 目錄或絕對路徑)。
    - `中文意譯`: 對應的中文標準答案。

    **範例 `metadata_train_fixed.csv`**:
    ```csv
    file,中文意譯
    train/audio1.wav,"這是一段測試語音"
    train/audio2.wav,"今天天氣真好"
    ```

### 2. 參數配置

打開 `finetune_Breeze_local.py` 腳本，修改頭部的基本配置變數：

```python
# ==============================================================================
# 基本配置 - 本機版本
# ==============================================================================

MODEL_ID = "MediaTek-Research/Breeze-ASR-25"
TRAIN_CSV = "metadata_train_fixed.csv"
TEST_CSV = "metadata_test_fixed.csv"
OUTPUT_DIR = "./breeze-asr-25-local"

# 本機訓練參數
QUICK_TEST_RATIO = 1  # 快速測試使用 100% 資料，可設為 0.1 (10%)
QUICK_MAX_STEPS = 5000  # 最大訓練步數
```

### 3. 執行微調

完成配置後，直接執行腳本：

```bash
python finetune_Breeze_local.py
```

腳本會自動處理環境設定、資料載入、模型訓練，並將訓練好的模型儲存到 `OUTPUT_DIR` 指定的目錄（預設為 `./breeze-asr-25-local`）。

---

## 第二部分：模型評估 (`evaluate_test_dataset.py`)

在微調完成後，使用此腳本來評估您的自訂模型在獨立測試集上的表現。

### 1. 資料準備

- **音訊檔案**: 準備一個包含所有測試音訊檔的資料夾。支援的格式包括 `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`。
- **參考文本 (CSV)**: 準備一個 CSV 檔案，包含音訊檔名和對應的標準答案。

    CSV 檔案必須包含以下兩欄：
    - `filename`: 音訊檔案的名稱。
    - `transcription`: 對應的中文標準答案。

    **範例 `test_data.csv`**:
    ```csv
    filename,transcription
    test1.wav,"這是第一個測試案例"
    test2.m4a,"這是第二個測試案例"
    ```
    您也可以加入 `domain`, `speaker_id` 等欄位進行更深入的分析。

### 2. 執行評估

此腳本支援透過命令列參數傳入配置，方便自動化測試。

**命令列參數說明**:

- `--model_path`: (必須) 您要評估的模型路徑。可以是本地路徑 (如 `./breeze-asr-25-local`) 或 Hugging Face Hub 上的模型名稱。
- `--test_csv`: (必須) 測試資料的 CSV 檔案路徑。
- `--test_audio`: (必須) 測試音訊所在的資料夾路徑。
- `--embedding_model`: (可選) 用於計算語義相似度的向量模型。預設為 `shibing624/text2vec-base-chinese`。
- `--output_dir`: (可選) 儲存評估報告的資料夾。預設為 `evaluation_results`。

**執行範例**:

```bash
python evaluate_test_dataset.py \
    --model_path ./breeze-asr-25-local \
    --test_csv ./path/to/your/test_data.csv \
    --test_audio ./path/to/your/audio_folder/
```

### 3. 查看結果

評估完成後，腳本會：

1.  **在終端機輸出摘要報告**:
    - 字詞錯誤率 (WER)
    - 平均/中位數語義相似度
    - 相似度分佈 (高/中/低)
    - 相似度最高和最低的範例

2.  **在輸出目錄 (`evaluation_results`) 生成檔案**:
    - `test_evaluation_YYYYMMDD_HHMMSS.json`: 包含所有評估資訊的完整報告。
    - `test_evaluation_YYYYMMDD_HHMMSS.csv`: 方便在 Excel 或其他工具中分析的詳細結果。
