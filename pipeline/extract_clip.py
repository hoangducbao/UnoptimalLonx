"""
pipeline/extract_clip.py -- Extract CLIP ViT-B/32 (512-d) image embeddings for ADL keyframes.

Usage:
    python -m pipeline.extract_clip [--batch-size 32] [--device cuda]
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

from . import config_adl as cfg


def load_model(model_id: str, device: str):
    print(f"[CLIP ViT-B/32] Loading model {model_id} on {device}...")
    model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


@torch.no_grad()
def extract_video_features(
    model,
    processor,
    keyframe_dir: Path,
    output_npy_path: Path,
    device: str,
    batch_size: int = 32,
):
    jpg_files = sorted(keyframe_dir.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
    if not jpg_files:
        return False

    all_feats = []
    for i in range(0, len(jpg_files), batch_size):
        batch_paths = jpg_files[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"[Warning] Corrupt image {p}: {e}")
                images.append(Image.new("RGB", (224, 224), color=0))

        inputs = processor(images=images, return_tensors="pt").to(device)
        out = model(**inputs)
        feats = out.image_embeds if hasattr(out, "image_embeds") else out[0]
        # Normalize L2
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.float().cpu().numpy())

    if all_feats:
        matrix = np.vstack(all_feats).astype("float16")
        output_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_npy_path, matrix)
        return True
    return False


def run(batch_size: int = cfg.BATCH_SIZE, device: str = cfg.DEVICE, overwrite: bool = False):
    cfg.ensure_directories()
    video_dirs = [p for p in sorted(cfg.KEYFRAME_DIR.iterdir()) if p.is_dir()]
    if not video_dirs:
        print(f"[Warning] No keyframe folders found in {cfg.KEYFRAME_DIR}")
        return

    model, processor = load_model(cfg.CLIP_VISION_MODEL_ID, device)

    print(f"[CLIP ViT-B/32] Processing {len(video_dirs)} videos...")
    for v_dir in tqdm(video_dirs, desc="Extracting CLIP"):
        video_id = v_dir.name
        out_npy = cfg.CLIP_EMBED_DIR / f"{video_id}.npy"
        if out_npy.exists() and not overwrite:
            continue

        extract_video_features(
            model=model,
            processor=processor,
            keyframe_dir=v_dir,
            output_npy_path=out_npy,
            device=device,
            batch_size=batch_size,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CLIP ViT-B/32 features for ADL keyframes")
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--device", type=str, default=cfg.DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run(batch_size=args.batch_size, device=args.device, overwrite=args.overwrite)
