# main_with_vocal_isolation.py
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal

import imageio_ffmpeg
import pandas as pd
from pydub import AudioSegment
from pytube import YouTube
# 核心處理套件
from youtube_transcript_api import (NoTranscriptFound, TranscriptsDisabled,
                                    YouTubeTranscriptApi)

# 設定 pydub 使用 imageio-ffmpeg 下載的 ffmpeg 可執行檔
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()


# --- 資料類別定義 (Data Class Definition) ---
@dataclass
class ClipperConfig:
    youtube_url: str
    output_dir: str = "output_from_subtitles"
    language_codes: List[str] = field(
        default_factory=lambda: ["zh-TW", "zh-Hant", "zh", "zh-Hans"]
    )
    output_format: Literal["audio", "video"] = "audio"
    perform_vocal_isolation: bool = True  # <<< 新增：是否執行人聲分離的開關


@dataclass
class ClippedSegment:
    index: int
    text: str
    start_time: float
    end_time: float
    duration: float
    original_clip_path: str  # <<< 修改：保留原始片段路徑
    processed_clip_path: str  # <<< 新增：指向最終處理過的檔案路徑（可能是純化後的人聲）


# --- 核心處理類別 (Core Processor Class) ---
class YouTubeSubtitleClipper:
    def __init__(self, config: ClipperConfig):
        self.config = config
        self.output_path = Path(self.config.output_dir)
        self.source_media_path: Path | None = None
        # <<< 修改：建立更詳細的目錄結構
        self.clips_path = self.output_path / "1_original_clips"
        self.isolated_path = self.output_path / "2_isolated_vocals"
        self.demucs_output_path = self.output_path / "temp_demucs_output"
        self.results: List[ClippedSegment] = []
        self.csv_rows: List[Dict[str, str]] = []  # 供輸出符合規格的 CSV

    def _ensure_dir_exists(self) -> None:
        self.output_path.mkdir(exist_ok=True)
        self.clips_path.mkdir(exist_ok=True)
        if self.config.perform_vocal_isolation:
            self.isolated_path.mkdir(exist_ok=True)
            self.demucs_output_path.mkdir(exist_ok=True)
        print(f"輸出目錄結構已建立於: {self.output_path}")

    # <<< 新增：執行人聲分離的函式
    def _isolate_vocals(self, audio_path: Path) -> Path | None:
        """
        使用 Demucs 對指定的音訊檔案進行人聲分離。
        返回純人聲音訊檔案的路徑。
        """
        print(f"  - [AI] 正在分離人聲: {audio_path.name}")
        try:
            # 動態偵測 GPU，並告訴 Demucs 使用對應裝置
            try:
                import torch  # 延遲載入，避免非必要依賴
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

            # Demucs 預設會在輸出目錄中建立一個名為 'htdemucs' 的子目錄
            model_name = "htdemucs"  # 這是 Demucs v4 的預設模型名稱
            expected_output_dir = self.demucs_output_path / model_name / audio_path.stem

            # 呼叫 Demucs 命令列工具
            command = [
                sys.executable,
                "-m",
                "demucs.separate",
                "-d",
                device,
                # 使用 4 stems（vocals、drums、bass、other）以獲得更乾淨的人聲
                "-o",
                str(self.demucs_output_path),
                str(audio_path),
            ]

            # 使用 subprocess.run，並捕獲輸出
            result = subprocess.run(command, check=True, capture_output=True, text=True)

            # 找到分離出的人聲檔案 (vocals.wav)
            vocals_file = expected_output_dir / "vocals.wav"
            if vocals_file.exists():
                # 將檔案移動並重新命名到我們的最終目錄
                final_vocal_path = self.isolated_path / f"{audio_path.stem}_vocals.wav"
                shutil.move(str(vocals_file), str(final_vocal_path))
                print(f"  - [成功] 人聲已儲存至: {final_vocal_path.name}")

                # 基於 ffmpeg 的頻域降噪與簡單濾波（移除低頻隆隆與高頻噪訊）
                try:
                    denoised_path = self.isolated_path / f"{audio_path.stem}_vocals_denoised.wav"
                    ffmpeg_exe = AudioSegment.converter
                    # 高通 -> FFT 降噪 -> 低通
                    af = "highpass=f=100,afftdn=nf=-25,lowpass=f=12000"
                    cmd = [
                        ffmpeg_exe,
                        "-y",
                        "-i",
                        str(final_vocal_path),
                        "-af",
                        af,
                        str(denoised_path),
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    print(f"  - [降噪] 已輸出：{denoised_path.name}")
                    return denoised_path
                except Exception as e:
                    print(f"  - [降噪略過] ffmpeg 降噪失敗：{e}")
                    return final_vocal_path
            else:
                print(f"  - [錯誤] Demucs 執行完畢，但找不到 vocals.wav 檔案。")
                print(f"  - Demucs stdout: {result.stdout}")
                print(f"  - Demucs stderr: {result.stderr}")
                return None
        except subprocess.CalledProcessError as e:
            print(f"  - [錯誤] Demucs 執行失敗。返回碼: {e.returncode}")
            print(f"  - stdout: {e.stdout}")
            print(f"  - stderr: {e.stderr}")
            return None
        except Exception as e:
            print(f"  - [錯誤] 執行人聲分離時發生未知錯誤: {e}")
            return None
        finally:
            # 清理 Demucs 產生的暫存資料夾
            if expected_output_dir.exists():
                shutil.rmtree(expected_output_dir.parent)

    def run(self) -> None:
        # ... (_get_transcript, _download_media 函式與之前相同，此處省略)
        # 完整的 _get_transcript, _download_media 函式請參考上一版本的回答
        self._ensure_dir_exists()
        transcript = self._get_transcript()
        if not transcript:
            return
        if not self._download_media() or not self.source_media_path:
            return

        print(f"\n--- 開始根據 {len(transcript)} 句字幕進行切割 ---")

        main_audio = AudioSegment.from_file(self.source_media_path)
        total_ms = len(main_audio)
        created_count = 0
        skipped_indices: List[int] = []

        for i, entry in enumerate(transcript):
            start_time, duration, text = (
                entry["start"],
                entry["duration"],
                entry["text"],
            )
            end_time = start_time + duration
            print(
                f"--- 正在處理片段 {i+1}/{len(transcript)} ({start_time:.2f}s - {end_time:.2f}s) ---"
            )
            print(f'  - 字幕: "{text[:40]}..."')

            # 安全邊界檢查與修正
            start_ms, end_ms = int(start_time * 1000), int(end_time * 1000)
            if start_ms >= total_ms:
                print("  - [跳過] 片段起始超出音訊長度。")
                skipped_indices.append(i)
                continue
            end_ms = min(end_ms, total_ms)
            if end_ms <= start_ms:
                print("  - [跳過] 片段長度無效。")
                skipped_indices.append(i)
                continue

            # 步驟 1: 切割原始音訊（輸出 wav）
            original_clip = main_audio[start_ms:end_ms]
            original_clip_path = self.clips_path / f"clip_{i:04d}.wav"
            original_clip.export(original_clip_path, format="wav")
            print(f"  - 原始片段已儲存至: {original_clip_path.name}")

            # 預設：CSV 取用的檔案路徑
            selected_output_path = original_clip_path

            # 步驟 2: (可選) 執行人聲分離（輸出 wav）
            processed_path = original_clip_path
            if self.config.perform_vocal_isolation:
                isolated_vocal_path = self._isolate_vocals(original_clip_path)
                if isolated_vocal_path:
                    processed_path = isolated_vocal_path
                    selected_output_path = isolated_vocal_path

            # 步驟 3: 記錄結果（內部詳單）
            self.results.append(
                ClippedSegment(
                    index=i,
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    original_clip_path=str(
                        original_clip_path.relative_to(self.output_path.parent)
                    ),
                    processed_clip_path=str(
                        Path(processed_path).relative_to(self.output_path.parent)
                    ),
                )
            )

            # 步驟 4: 建立 CSV 行（需求欄位：漢字、檔案位置）
            self.csv_rows.append(
                {
                    "漢字": text,
                    "檔案位置": str(Path(selected_output_path).resolve()),
                }
            )
            created_count += 1

        self._export_to_csv()
        # 匹配性檢查
        if created_count == len(transcript):
            print(f"\n✅ 所有字幕均已對應到音檔：{created_count}/{len(transcript)}")
        else:
            print(
                f"\n⚠️ 有部分字幕未能對應到音檔：{created_count}/{len(transcript)}，跳過索引：{skipped_indices}"
            )

    # ... (_get_transcript, _download_media, _export_to_csv 函式與之前相同)
    # 為了簡潔，此處省略，請從上一版本複製過來
    def _get_transcript(self) -> List[Dict[str, Any]] | None:
        try:
            video_id = self.config.youtube_url.split("v=")[1].split("&")[0]
            print(f"正在為影片 ID: {video_id} 尋找字幕...")
            ytt_api = YouTubeTranscriptApi()
            fetched_transcript = ytt_api.fetch(
                video_id, languages=self.config.language_codes
            )
            transcript = fetched_transcript.to_raw_data()
            print(f"成功獲取 {len(transcript)} 句字幕。")
            return transcript
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            print(f"錯誤：無法獲取此影片的字幕。 ({e})")
            return None
        except Exception as e:
            print(f"獲取字幕時發生未知錯誤: {e}")
            return None

    def _download_media(self) -> bool:
        try:
            print(f"正在從 YouTube 下載來源媒體: {self.config.youtube_url}")

            # 僅保留 v 參數，避免 playlist 或其他參數造成 400
            url = self.config.youtube_url
            if "v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
                url = f"https://www.youtube.com/watch?v={video_id}"

            # 預設目標（若實際副檔名不同，稍後會自動偵測）
            self.source_media_path = self.output_path / "source_audio.m4a"
            if any(self.output_path.glob("source_audio.*")):
                print("來源媒體檔案已存在，跳過下載。")
                return True

            # 方案 A: 先嘗試 pytube 純音訊
            try:
                yt = YouTube(url)
                stream = yt.streams.filter(only_audio=True, mime_type="audio/mp4").first() or yt.streams.get_audio_only()
                if stream is None:
                    raise RuntimeError("找不到可用的音訊串流。")
                tmp_name = "source_audio_tmp"
                stream.download(output_path=str(self.output_path), filename=tmp_name)
                # 推測副檔名
                guessed_ext = stream.mime_type.split("/")[-1] if stream.mime_type else "m4a"
                tmp_path = self.output_path / f"{tmp_name}.{guessed_ext}"
                if tmp_path.exists():
                    tmp_path.replace(self.source_media_path)
                print(f"來源媒體已成功下載至: {self.source_media_path}")
                return True
            except Exception as e:
                print(f"pytube 下載失敗，改用 yt-dlp 後備方案。原因: {e}")

            # 方案 B: 後備使用 yt-dlp 下載最佳音訊（優先選 m4a，無需轉檔以避免依賴 ffmpeg）
            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "yt_dlp",
                    "-f",
                    "ba[ext=m4a]/bestaudio/best",
                    "-o",
                    str(self.output_path / "source_audio.%(ext)s"),
                    url,
                ]
                subprocess.run(cmd, check=True, capture_output=True)

                # 下載後自動偵測實際副檔名
                downloaded = list(self.output_path.glob("source_audio.*"))
                if not downloaded:
                    raise RuntimeError("yt-dlp 執行成功但未找到輸出檔案。")
                self.source_media_path = downloaded[0]
                print(f"來源媒體已成功下載至: {self.source_media_path}")
                return True
            except Exception as e:
                print(f"yt-dlp 下載失敗: {e}")
                return False
        except Exception as e:
            print(f"下載媒體時發生錯誤: {e}")
            return False

    def _export_to_csv(self) -> None:
        # 依需求輸出：欄位為「漢字」、「檔案位置」，使用 self.csv_rows
        if not self.csv_rows:
            print("沒有可匯出的資料。")
            return
        df = pd.DataFrame(self.csv_rows, columns=["漢字", "檔案位置"])
        csv_path = self.output_path / "clips_mapping.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n處理完成！所有資料已儲存至: {csv_path}")


# --- 程式進入點 (Entry Point) ---
if __name__ == "__main__":
    TARGET_URL = "https://www.youtube.com/watch?v=mL1K-Wcf3qo&list=PL_OjiqlIsqlK1NlhFzQ8ycOLPj0NWHDK1&ab_channel=%E6%B0%91%E8%A6%96%E6%88%B2%E5%8A%87%E9%A4%A8FormosaTVDramas"

    config = ClipperConfig(
        youtube_url=TARGET_URL,
        output_dir="youtube_clips_isolated",
        output_format="audio",  # 人聲分離只對 audio 有意義
        perform_vocal_isolation=True,  # <<< 設定為 True 來啟動 AI 去背景音
    )

    # 警告使用者 CPU 會很慢
    if config.perform_vocal_isolation:
        try:
            import torch

            if not torch.cuda.is_available():
                print("\n" + "=" * 50)
                print("警告：未偵測到 CUDA GPU。Demucs 將在 CPU 上執行。")
                print("      處理速度將會非常非常緩慢。")
                print("=" * 50 + "\n")
        except ImportError:
            pass  # 如果連 torch 都沒有，demucs-cpu 會處理

    clipper = YouTubeSubtitleClipper(config)
    clipper.run()
