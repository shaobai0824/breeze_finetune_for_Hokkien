#!/usr/bin/env python3
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
from youtube_transcript_api import (NoTranscriptFound, TranscriptsDisabled,
                                    YouTubeTranscriptApi)

# 指定 ffmpeg
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class ClipperConfig:
    youtube_url: str
    output_dir: str = "output_from_subtitles_v4"
    language_codes: List[str] = field(
        default_factory=lambda: ["zh-TW", "zh-Hant", "zh", "zh-Hans"]
    )
    perform_vocal_isolation: bool = True
    demucs_model: Literal["htdemucs", "mdx_extra", "mdx_extra_q"] = "htdemucs"


class YouTubeSubtitleClipperV4:
    def __init__(self, config: ClipperConfig):
        self.config = config
        self.output_path = Path(self.config.output_dir)
        self.clips_path = self.output_path / "1_original_clips"
        self.isolated_path = self.output_path / "2_isolated_vocals"
        self.demucs_output_path = self.output_path / "temp_demucs_output"
        self.csv_rows: List[Dict[str, str]] = []

    def _ensure_dir_exists(self) -> None:
        self.output_path.mkdir(exist_ok=True)
        self.clips_path.mkdir(exist_ok=True)
        if self.config.perform_vocal_isolation:
            self.isolated_path.mkdir(exist_ok=True)
            self.demucs_output_path.mkdir(exist_ok=True)

    def _get_transcript(self) -> List[Dict[str, Any]] | None:
        try:
            video_id = self.config.youtube_url.split("v=")[1].split("&")[0]
            ytt_api = YouTubeTranscriptApi()
            fetched = ytt_api.fetch(video_id, languages=self.config.language_codes)
            return fetched.to_raw_data()
        except (TranscriptsDisabled, NoTranscriptFound):
            return None

    def _download_media(self) -> Path | None:
        url = self.config.youtube_url
        if "v=" in url:
            vid = url.split("v=")[1].split("&")[0]
            url = f"https://www.youtube.com/watch?v={vid}"
        target = self.output_path / "source_audio.m4a"
        if any(self.output_path.glob("source_audio.*")):
            return next(self.output_path.glob("source_audio.*"))
        try:
            yt = YouTube(url)
            stream = yt.streams.filter(only_audio=True, mime_type="audio/mp4").first() or yt.streams.get_audio_only()
            tmp = self.output_path / "source_audio_tmp.m4a"
            stream.download(output_path=str(self.output_path), filename="source_audio_tmp")
            guess = stream.mime_type.split("/")[-1] if stream.mime_type else "m4a"
            tmp_real = self.output_path / f"source_audio_tmp.{guess}"
            if tmp_real.exists():
                tmp_real.replace(target)
            return target
        except Exception:
            pass
        # fallback: yt-dlp
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
            files = list(self.output_path.glob("source_audio.*"))
            return files[0] if files else None
        except Exception:
            return None

    def _isolate_vocals(self, audio_path: Path) -> Path | None:
        try:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

            model = self.config.demucs_model
            expected = self.demucs_output_path / model / audio_path.stem / "vocals.wav"
            cmd = [
                sys.executable,
                "-m",
                "demucs.separate",
                "-d",
                device,
                "-n",
                model,
                "-o",
                str(self.demucs_output_path),
                str(audio_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if expected.exists():
                final_vocals = self.isolated_path / f"{audio_path.stem}_{model}_vocals.wav"
                shutil.move(str(expected), str(final_vocals))
                # 簡單後處理
                try:
                    ffmpeg = AudioSegment.converter
                    out = self.isolated_path / f"{audio_path.stem}_{model}_vocals_denoised.wav"
                    af = "highpass=f=100,afftdn=nf=-25,lowpass=f=12000"
                    subprocess.run([
                        ffmpeg, "-y", "-i", str(final_vocals), "-af", af, str(out)
                    ], check=True, capture_output=True)
                    return out
                except Exception:
                    return final_vocals
            return None
        except Exception:
            return None
        finally:
            # 清理臨時目錄
            model = self.config.demucs_model
            base = self.demucs_output_path / model
            if base.exists():
                shutil.rmtree(base)

    def run(self) -> None:
        self._ensure_dir_exists()
        t = self._get_transcript()
        if not t:
            return
        src = self._download_media()
        if not src:
            return
        audio = AudioSegment.from_file(src)
        L = len(audio)
        created = 0
        for i, e in enumerate(t):
            s, d, text = e["start"], e["duration"], e["text"]
            st, ed = int(s * 1000), int((s + d) * 1000)
            if st >= L:
                continue
            ed = min(ed, L)
            if ed <= st:
                continue
            seg = audio[st:ed]
            raw_path = self.clips_path / f"clip_{i:04d}.wav"
            seg.export(raw_path, format="wav")
            final_path = raw_path
            if self.config.perform_vocal_isolation:
                v = self._isolate_vocals(raw_path)
                if v:
                    final_path = v
            self.csv_rows.append({"漢字": text, "檔案位置": str(Path(final_path).resolve())})
            created += 1
        df = pd.DataFrame(self.csv_rows, columns=["漢字", "檔案位置"])
        out_csv = self.output_path / "clips_mapping.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    TARGET_URL = "https://www.youtube.com/watch?v=mL1K-Wcf3qo"
    cfg = ClipperConfig(
        youtube_url=TARGET_URL,
        output_dir="youtube_clips_v4_models",
        perform_vocal_isolation=True,
        demucs_model="mdx_extra",  # 可改為 "htdemucs" 或 "mdx_extra_q"
    )
    YouTubeSubtitleClipperV4(cfg).run()


