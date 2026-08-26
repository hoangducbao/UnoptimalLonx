"""
pipeline/extract_summary.py -- Summarize videos from captions & transcripts, then embed summaries with SigLIP2.

Generates:
  1. ADLDataExtracted/summaries/{video_id}.txt
  2. ADLDataExtracted/summary_embed/{video_id}.npy (1 x 768 vector)

Usage:
    python -m pipeline.extract_summary [--device cuda]
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from . import config_adl as cfg


def create_summary_text(video_id: str, cap_csv: Path, asr_csv: Path) -> str:
    parts = []

    if cap_csv.exists():
        df_cap = pd.read_csv(cap_csv)
        if not df_cap.empty and "caption_text" in df_cap.columns:
            # Sample or take key event descriptions
            caps = df_cap["caption_text"].dropna().tolist()
            # Deduplicate adjacent repetitive captions
            dedup_caps = []
            for c in caps:
                c_clean = c.strip()
                if not dedup_caps or c_clean.lower() != dedup_caps[-1].lower():
                    dedup_caps.append(c_clean)
            if dedup_caps:
                parts.append("Visual action sequence: " + " -> ".join(dedup_caps[:25]) + ".")

    if asr_csv.exists():
        df_asr = pd.read_csv(asr_csv)
        if not df_asr.empty and "text" in df_asr.columns:
            transcripts = df_asr["text"].dropna().tolist()
            if transcripts:
                parts.append("Spoken audio: " + " ".join(transcripts[:30]))

    if not parts:
        return f"Daily life video recording of activities in {video_id}."

    return f"Activity video {video_id}. " + " ".join(parts)


def run(device: str = cfg.DEVICE, overwrite: bool = False):
    cfg.ensure_directories()
    # List all videos from keyframe directories
    video_dirs = [p for p in sorted(cfg.KEYFRAME_DIR.iterdir()) if p.is_dir()]
    if not video_dirs:
        # Check raw videos if keyframes not yet extracted
        video_dirs = [p for p in sorted(cfg.VIDEO_DIR.glob("*.mp4"))]

    if not video_dirs:
        print(f"[Warning] No videos found to summarize.")
        return

    print(f"[Summary] Loading SigLIP2 text encoder on {device}...")
    siglip_model = AutoModel.from_pretrained(cfg.SIGLIP2_MODEL_ID).to(device).eval()
    siglip_processor = AutoProcessor.from_pretrained(cfg.SIGLIP2_MODEL_ID)

    print(f"[Summary] Generating summaries and embeddings for {len(video_dirs)} videos...")
    for v_item in tqdm(video_dirs, desc="Video Summaries"):
        video_id = v_item.stem if v_item.is_file() else v_item.name
        txt_path = cfg.SUMMARY_DIR / f"{video_id}.txt"
        npy_path = cfg.SUMMARY_EMBED_DIR / f"{video_id}.npy"

        if txt_path.exists() and npy_path.exists() and not overwrite:
            continue

        cap_csv = cfg.CAPTION_DIR / f"{video_id}.csv"
        asr_csv = cfg.ASR_TRANSCRIPT_DIR / f"{video_id}.csv"

        summary_text = create_summary_text(video_id, cap_csv, asr_csv)

        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(summary_text, encoding="utf-8")

        inputs = siglip_processor(text=[summary_text], padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = siglip_model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        vec = feats.float().cpu().numpy().astype("float32")

        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, vec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video summaries and SigLIP2 embeddings for ADL")
    parser.add_argument("--device", type=str, default=cfg.DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run(device=args.device, overwrite=args.overwrite)
