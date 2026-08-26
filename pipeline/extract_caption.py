"""
pipeline/extract_caption.py -- Generate keyframe dense captions and embed them with SigLIP2 text tower.

Generates:
  1. ADLDataExtracted/captions/{video_id}.csv: [video_id, frame_id, caption_text]
  2. ADLDataExtracted/caption_embed/{video_id}.npy (SigLIP2 text embeddings)
  3. ADLDataExtracted/caption_embed/{video_id}.csv [frame_id, text]

Usage:
    python -m pipeline.extract_caption [--model-type blip|florence2|qwen2_vl] [--device cuda]
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from . import config_adl as cfg


def load_caption_model(model_type: str, device: str):
    print(f"[Caption] Loading captioning model ({model_type}) on {device}...")
    if model_type == "florence2":
        model_id = "microsoft/Florence-2-base"
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return model, processor, "florence2"
    elif model_type == "blip":
        from transformers import BlipForConditionalGeneration, BlipProcessor
        model_id = "Salesforce/blip-image-captioning-large"
        model = BlipForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        processor = BlipProcessor.from_pretrained(model_id)
        return model, processor, "blip"
    else:  # fallback fast BLIP base
        from transformers import BlipForConditionalGeneration, BlipProcessor
        model_id = "Salesforce/blip-image-captioning-base"
        model = BlipForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        processor = BlipProcessor.from_pretrained(model_id)
        return model, processor, "blip"


@torch.no_grad()
def generate_captions_for_video(
    model,
    processor,
    model_type: str,
    keyframe_dir: Path,
    video_id: str,
    device: str,
    batch_size: int = 16,
) -> pd.DataFrame:
    jpg_files = sorted(keyframe_dir.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
    if not jpg_files:
        return pd.DataFrame()

    rows = []
    for i in range(0, len(jpg_files), batch_size):
        batch_paths = jpg_files[i : i + batch_size]
        images = []
        frame_ids = []
        for p in batch_paths:
            try:
                frame_idx = int(p.stem) - 1 if p.stem.isdigit() else len(rows)
                img = Image.open(p).convert("RGB")
                images.append(img)
                frame_ids.append(frame_idx)
            except Exception as e:
                print(f"[Warning] Failed to read {p}: {e}")

        if not images:
            continue

        if model_type == "florence2":
            prompt = "<MORE_DETAILED_CAPTION>"
            inputs = processor(text=[prompt] * len(images), images=images, return_tensors="pt").to(device)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=3,
            )
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
            for fid, gen_text in zip(frame_ids, generated_texts):
                parsed = processor.post_process_generation(gen_text, task=prompt, image_size=(images[0].width, images[0].height))
                cap = parsed.get(prompt, "").strip()
                rows.append({"video_id": video_id, "frame_id": fid, "caption_text": cap})
        else:  # BLIP
            inputs = processor(images=images, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=60)
            captions = processor.batch_decode(out, skip_special_tokens=True)
            for fid, cap in zip(frame_ids, captions):
                rows.append({"video_id": video_id, "frame_id": fid, "caption_text": cap.strip()})

    return pd.DataFrame(rows)


def embed_captions_siglip2(
    caption_df: pd.DataFrame,
    siglip_model,
    siglip_processor,
    out_npy: Path,
    out_csv: Path,
    device: str,
    batch_size: int = 64,
):
    if caption_df.empty:
        return

    texts = caption_df["caption_text"].tolist()
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

        # Save metadata CSV for SigLIP caption index
        meta_df = pd.DataFrame({
            "frame_id": caption_df["frame_id"],
            "text": caption_df["caption_text"]
        })
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        meta_df.to_csv(out_csv, index=False)


def run(model_type: str = "blip", device: str = cfg.DEVICE, overwrite: bool = False):
    cfg.ensure_directories()
    video_dirs = [p for p in sorted(cfg.KEYFRAME_DIR.iterdir()) if p.is_dir()]
    if not video_dirs:
        print(f"[Warning] No keyframe folders found in {cfg.KEYFRAME_DIR}")
        return

    cap_model, cap_processor, m_type = load_caption_model(model_type, device)

    # Also load SigLIP2 text model to generate caption embeddings
    print(f"[Caption] Loading SigLIP2 text encoder for embedding generation...")
    siglip_model = AutoModel.from_pretrained(cfg.SIGLIP2_MODEL_ID).to(device).eval()
    siglip_processor = AutoProcessor.from_pretrained(cfg.SIGLIP2_MODEL_ID)

    print(f"[Caption] Processing captions for {len(video_dirs)} videos...")
    for v_dir in tqdm(video_dirs, desc="Captions & Embeddings"):
        video_id = v_dir.name
        cap_csv_path = cfg.CAPTION_DIR / f"{video_id}.csv"
        embed_npy_path = cfg.CAPTION_EMBED_DIR / f"{video_id}.npy"
        embed_csv_path = cfg.CAPTION_EMBED_DIR / f"{video_id}.csv"

        if cap_csv_path.exists() and embed_npy_path.exists() and not overwrite:
            continue

        df = generate_captions_for_video(
            model=cap_model,
            processor=cap_processor,
            model_type=m_type,
            keyframe_dir=v_dir,
            video_id=video_id,
            device=device,
        )

        if not df.empty:
            df.to_csv(cap_csv_path, index=False)
            embed_captions_siglip2(
                caption_df=df,
                siglip_model=siglip_model,
                siglip_processor=siglip_processor,
                out_npy=embed_npy_path,
                out_csv=embed_csv_path,
                device=device,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract captions and embeddings for ADL keyframes")
    parser.add_argument("--model-type", type=str, default="blip", choices=["blip", "florence2"])
    parser.add_argument("--device", type=str, default=cfg.DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run(model_type=args.model_type, device=args.device, overwrite=args.overwrite)
