"""
pipeline/extract_asr.py -- Transcribe audio using Whisper and map segments to keyframes.

Generates:
  1. ADLDataExtracted/transcripts/{video_id}.csv: [segment_id, start_sec, end_sec, text]
  2. ADLDataExtracted/transcript_embed/{video_id}.npy (SigLIP2 text embeddings)
  3. ADLDataExtracted/transcript_embed/{video_id}.csv [frame_id, segment_id, start_sec, text]

Usage:
    python -m pipeline.extract_asr [--model-size base|medium|large-v3] [--device cuda]
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from . import config_adl as cfg


def transcribe_video(whisper_model, video_path: Path):
    try:
        result = whisper_model.transcribe(str(video_path), verbose=False)
        segments = result.get("segments", [])
        rows = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if text:
                rows.append({
                    "segment_id": int(seg.get("id", len(rows))),
                    "start_sec": round(float(seg.get("start", 0.0)), 2),
                    "end_sec": round(float(seg.get("end", 0.0)), 2),
                    "text": text
                })
        return pd.DataFrame(rows)
    except Exception as e:
        # Video might have no audio stream
        return pd.DataFrame(columns=["segment_id", "start_sec", "end_sec", "text"])


def map_segments_to_keyframes(transcript_df: pd.DataFrame, map_csv_path: Path) -> pd.DataFrame:
    if transcript_df.empty or not map_csv_path.exists():
        return pd.DataFrame()

    map_df = pd.read_csv(map_csv_path)
    if map_df.empty:
        return pd.DataFrame()

    mapped_rows = []
    for _, r in transcript_df.iterrows():
        start_t = float(r["start_sec"])
        # Find closest keyframe by pts_time
        idx = (map_df["pts_time"] - start_t).abs().idxmin()
        frame_id = int(map_df.loc[idx, "n"]) - 1  # 0-indexed frame_id

        mapped_rows.append({
            "frame_id": frame_id,
            "segment_id": int(r["segment_id"]),
            "start_sec": start_t,
            "text": r["text"]
        })

    return pd.DataFrame(mapped_rows)


def embed_asr_siglip2(
    mapped_df: pd.DataFrame,
    siglip_model,
    siglip_processor,
    out_npy: Path,
    out_csv: Path,
    device: str,
    batch_size: int = 64,
):
    if mapped_df.empty:
        return

    texts = mapped_df["text"].tolist()
    all_vecs = []

    for i in range(0, len(texts), batch_size):
        b_texts = texts[i : i + batch_size]
        inputs = siglip_processor(text=b_texts, padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = siglip_model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        all_vecs.append(feats.float().cpu().numpy())

    if all_vecs:
        matrix = np.vstack(all_vecs).astype("float32")
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, matrix)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        mapped_df.to_csv(out_csv, index=False)


def run(model_size: str = "base", device: str = cfg.DEVICE, overwrite: bool = False):
    cfg.ensure_directories()
    video_files = sorted(list(cfg.VIDEO_DIR.glob("*.mp4")) + list(cfg.VIDEO_DIR.glob("*.mkv")) + list(cfg.VIDEO_DIR.glob("*.avi")))
    if not video_files:
        print(f"[Warning] No video files found in {cfg.VIDEO_DIR}")
        return

    import whisper
    print(f"[ASR] Loading Whisper ({model_size}) on {device}...")
    whisper_model = whisper.load_model(model_size, device=device)

    print(f"[ASR] Loading SigLIP2 text encoder...")
    siglip_model = AutoModel.from_pretrained(cfg.SIGLIP2_MODEL_ID).to(device).eval()
    siglip_processor = AutoProcessor.from_pretrained(cfg.SIGLIP2_MODEL_ID)

    print(f"[ASR] Transcribing audio for {len(video_files)} videos...")
    for v_path in tqdm(video_files, desc="ASR Transcription"):
        video_id = v_path.stem
        raw_csv_path = cfg.ASR_TRANSCRIPT_DIR / f"{video_id}.csv"
        embed_npy_path = cfg.ASR_EMBED_DIR / f"{video_id}.npy"
        embed_csv_path = cfg.ASR_EMBED_DIR / f"{video_id}.csv"
        map_csv_path = cfg.MAP_KEYFRAMES_DIR / f"{video_id}.csv"

        if raw_csv_path.exists() and embed_npy_path.exists() and not overwrite:
            continue

        raw_df = transcribe_video(whisper_model, v_path)
        if not raw_df.empty:
            raw_df.to_csv(raw_csv_path, index=False)
            mapped_df = map_segments_to_keyframes(raw_df, map_csv_path)
            if not mapped_df.empty:
                embed_asr_siglip2(
                    mapped_df=mapped_df,
                    siglip_model=siglip_model,
                    siglip_processor=siglip_processor,
                    out_npy=embed_npy_path,
                    out_csv=embed_csv_path,
                    device=device,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ASR transcripts and embeddings for ADL videos")
    parser.add_argument("--model-size", type=str, default="base", choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--device", type=str, default=cfg.DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run(model_size=args.model_size, device=args.device, overwrite=args.overwrite)
