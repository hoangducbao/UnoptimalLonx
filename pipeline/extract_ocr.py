"""
pipeline/extract_ocr.py -- Extract text from keyframes using OCR (EasyOCR / PaddleOCR).

Generates:
  ADLDataExtracted/ocr/{video_id}.csv with columns: [frame_id, text]

Usage:
    python -m pipeline.extract_ocr [--engine easyocr|paddleocr] [--gpu]
"""

import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

from . import config_adl as cfg


def get_ocr_engine(engine_name: str, use_gpu: bool):
    print(f"[OCR] Initializing OCR engine ({engine_name})...")
    if engine_name == "easyocr":
        import easyocr
        reader = easyocr.Reader(["en", "vi"], gpu=use_gpu)
        return reader, "easyocr"
    elif engine_name == "paddleocr":
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=use_gpu)
        return ocr, "paddleocr"
    else:
        # Fallback to easyocr
        import easyocr
        reader = easyocr.Reader(["en"], gpu=use_gpu)
        return reader, "easyocr"


def process_video_ocr(engine, engine_type: str, keyframe_dir: Path, out_csv: Path):
    jpg_files = sorted(keyframe_dir.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
    if not jpg_files:
        return

    rows = []
    for p in jpg_files:
        frame_idx = int(p.stem) - 1 if p.stem.isdigit() else len(rows)
        detected_texts = []
        try:
            if engine_type == "easyocr":
                results = engine.readtext(str(p), detail=0)
                detected_texts = [t.strip() for t in results if t.strip()]
            elif engine_type == "paddleocr":
                result = engine.ocr(str(p), cls=True)
                if result and result[0]:
                    detected_texts = [line[1][0].strip() for line in result[0] if line[1][0].strip()]
        except Exception as e:
            print(f"[Warning] OCR failed on {p}: {e}")

        if detected_texts:
            full_text = " ".join(detected_texts)
            rows.append({"frame_id": frame_idx, "text": full_text})

    if rows:
        df = pd.DataFrame(rows)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)


def run(engine_name: str = "easyocr", use_gpu: bool = (cfg.DEVICE == "cuda"), overwrite: bool = False):
    cfg.ensure_directories()
    video_dirs = [p for p in sorted(cfg.KEYFRAME_DIR.iterdir()) if p.is_dir()]
    if not video_dirs:
        print(f"[Warning] No keyframe folders found in {cfg.KEYFRAME_DIR}")
        return

    engine, engine_type = get_ocr_engine(engine_name, use_gpu)

    print(f"[OCR] Running OCR on {len(video_dirs)} videos...")
    for v_dir in tqdm(video_dirs, desc="OCR Extraction"):
        video_id = v_dir.name
        out_csv = cfg.OCR_DIR / f"{video_id}.csv"
        if out_csv.exists() and not overwrite:
            continue

        process_video_ocr(engine, engine_type, v_dir, out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract OCR text from ADL keyframes")
    parser.add_argument("--engine", type=str, default="easyocr", choices=["easyocr", "paddleocr"])
    parser.add_argument("--gpu", action="store_true", default=(cfg.DEVICE == "cuda"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run(engine_name=args.engine, use_gpu=args.gpu, overwrite=args.overwrite)
