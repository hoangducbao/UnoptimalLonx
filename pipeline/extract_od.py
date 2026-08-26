"""
pipeline/extract_od.py -- Extract Object Detection (OD) labels from keyframes using YOLO / Torchvision.

Generates:
  1. ADLDataExtracted/filtered_object/{video_id}.csv with columns: [keyframe_id, class_name, score]
  2. Builds or updates ADLDataExtracted/filtered_object/class_vocab.csv

Usage:
    python -m pipeline.extract_od [--model yolov8x.pt] [--conf 0.25] [--device cuda]
"""

import argparse
import csv
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

from . import config_adl as cfg


def load_detector(model_name: str, device: str):
    """
    Loads YOLO detector via Ultralytics (or falls back to torchvision FasterRCNN).
    """
    print(f"[Object Detection] Loading model {model_name} on {device}...")
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        return model, "ultralytics"
    except ImportError:
        print("[Object Detection] 'ultralytics' not installed. Falling back to torchvision Faster R-CNN...")
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
        categories = weights.meta["categories"]
        return (model, categories), "torchvision"


def process_video_od(
    detector,
    detector_type: str,
    keyframe_dir: Path,
    out_csv: Path,
    conf_threshold: float = 0.25,
    device: str = "cuda",
    batch_size: int = 16,
):
    jpg_files = sorted(keyframe_dir.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
    if not jpg_files:
        return set()

    rows = []
    classes_found = set()

    if detector_type == "ultralytics":
        for i in range(0, len(jpg_files), batch_size):
            batch_paths = [str(p) for p in jpg_files[i : i + batch_size]]
            results = detector.predict(batch_paths, conf=conf_threshold, device=device, verbose=False)
            for p, res in zip(jpg_files[i : i + batch_size], results):
                keyframe_id = int(p.stem) if p.stem.isdigit() else 1
                if res.boxes is not None and len(res.boxes) > 0:
                    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                    confs = res.boxes.conf.cpu().numpy()
                    names = res.names
                    for c_id, conf in zip(cls_ids, confs):
                        c_name = names.get(c_id, f"class_{c_id}").strip().lower()
                        classes_found.add(c_name)
                        rows.append({
                            "keyframe_id": keyframe_id,
                            "class_name": c_name,
                            "score": round(float(conf), 4),
                        })
    else:  # torchvision
        import torchvision.transforms.functional as TF
        model, categories = detector
        for p in jpg_files:
            keyframe_id = int(p.stem) if p.stem.isdigit() else 1
            try:
                img = Image.open(p).convert("RGB")
                img_t = TF.to_tensor(img).to(device)
                with torch.no_grad():
                    preds = model([img_t])[0]
                scores = preds["scores"].cpu().numpy()
                labels = preds["labels"].cpu().numpy()
                for score, label_id in zip(scores, labels):
                    if score >= conf_threshold and label_id < len(categories):
                        c_name = categories[label_id].strip().lower()
                        classes_found.add(c_name)
                        rows.append({
                            "keyframe_id": keyframe_id,
                            "class_name": c_name,
                            "score": round(float(score), 4),
                        })
            except Exception as e:
                print(f"[Warning] OD error on {p}: {e}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
        # Deduplicate identical (keyframe_id, class_name) keeping highest score
        df = df.sort_values("score", ascending=False).drop_duplicates(subset=["keyframe_id", "class_name"]).sort_values(["keyframe_id", "score"], ascending=[True, False])
        df.to_csv(out_csv, index=False)
    else:
        pd.DataFrame(columns=["keyframe_id", "class_name", "score"]).to_csv(out_csv, index=False)

    return classes_found


def build_class_vocab_from_dir(od_dir: Path, vocab_csv: Path):
    """Scan all OD CSVs and write class_vocab.csv."""
    csv_paths = sorted(p for p in od_dir.glob("*.csv") if p.name != "class_vocab.csv")
    if not csv_paths:
        return
    names = set()
    for path in csv_paths:
        try:
            df = pd.read_csv(path, usecols=["class_name"])
            for name in df["class_name"].dropna():
                clean = " ".join(str(name).strip().lower().split())
                if clean:
                    names.add(clean)
        except Exception:
            continue
    vocab_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name"])
        for name in sorted(names):
            writer.writerow([name])
    print(f"[Vocab] Wrote {len(names)} unique classes to {vocab_csv}")


def run(
    model_name: str = cfg.YOLO_MODEL_ID,
    conf: float = 0.25,
    device: str = cfg.DEVICE,
    batch_size: int = cfg.BATCH_SIZE,
    overwrite: bool = False,
):
    cfg.ensure_directories()
    video_dirs = [p for p in sorted(cfg.KEYFRAME_DIR.iterdir()) if p.is_dir()]
    if not video_dirs:
        print(f"[Warning] No keyframe folders found in {cfg.KEYFRAME_DIR}")
        return

    detector, detector_type = load_detector(model_name, device)
    print(f"[Object Detection] Processing {len(video_dirs)} videos with {model_name}...")

    all_classes = set()
    for v_dir in tqdm(video_dirs, desc="Extracting Object Detections"):
        video_id = v_dir.name
        out_csv = cfg.FILTERED_OBJECT_DIR / f"{video_id}.csv"
        if out_csv.exists() and not overwrite:
            continue

        c_set = process_video_od(
            detector=detector,
            detector_type=detector_type,
            keyframe_dir=v_dir,
            out_csv=out_csv,
            conf_threshold=conf,
            device=device,
            batch_size=batch_size,
        )
        all_classes.update(c_set)

    build_class_vocab_from_dir(cfg.FILTERED_OBJECT_DIR, cfg.CLASS_VOCAB_CSV)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Object Detection labels from keyframes")
    parser.add_argument("--model", type=str, default=cfg.YOLO_MODEL_ID)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=cfg.DEVICE)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run(
        model_name=args.model,
        conf=args.conf,
        device=args.device,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
