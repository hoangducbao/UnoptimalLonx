"""
pipeline/run_adl_pipeline.py -- Unified execution pipeline for ADL dataset processing.

Runs all or selected feature extraction stages for the ADL dataset.

Examples:
    # Run full end-to-end pipeline:
    python -m pipeline.run_adl_pipeline --all

    # Run specific stages:
    python -m pipeline.run_adl_pipeline --keyframes --siglip --clip
    python -m pipeline.run_adl_pipeline --caption --ocr
"""

import argparse
import sys
from pathlib import Path

from . import config_adl as cfg


def main():
    parser = argparse.ArgumentParser(description="End-to-end multi-modal extraction pipeline for ADL video dataset")
    parser.add_argument("--all", action="store_true", help="Run all extraction stages end-to-end")
    parser.add_argument("--keyframes", action="store_true", help="Stage 1: Extract keyframes & map-keyframes CSVs")
    parser.add_argument("--siglip", action="store_true", help="Stage 2: Extract SigLIP2 image embeddings")
    parser.add_argument("--clip", action="store_true", help="Stage 3: Extract CLIP ViT-B/32 image embeddings")
    parser.add_argument("--caption", action="store_true", help="Stage 4: Generate Dense Captions & caption embeddings")
    parser.add_argument("--ocr", action="store_true", help="Stage 5: Extract OCR text")
    parser.add_argument("--asr", action="store_true", help="Stage 6: Transcribe speech with Whisper & embed")
    parser.add_argument("--summary", action="store_true", help="Stage 7: Generate video summaries & embeddings")
    parser.add_argument("--od", action="store_true", help="Stage 8: Extract Object Detection labels & class vocabulary")

    # Options
    parser.add_argument("--sample-fps", type=float, default=1.0, help="FPS sampling rate for keyframes")
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE, help="Batch size for neural models")
    parser.add_argument("--device", type=str, default=cfg.DEVICE, help="cuda or cpu")
    parser.add_argument("--caption-model", type=str, default="blip", choices=["blip", "florence2"])
    parser.add_argument("--ocr-engine", type=str, default="easyocr", choices=["easyocr", "paddleocr"])
    parser.add_argument("--whisper-size", type=str, default="base", choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--yolo-model", type=str, default=cfg.YOLO_MODEL_ID, help="YOLO model weights file/name")
    parser.add_argument("--od-conf", type=float, default=0.25, help="Confidence threshold for Object Detection")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extractions")

    args = parser.parse_args()

    # If no flags passed, show help
    if not (args.all or args.keyframes or args.siglip or args.clip or args.caption or args.ocr or args.asr or args.summary or args.od):
        parser.print_help()
        print("\n[Error] Please specify --all or at least one stage (e.g. --keyframes, --siglip, --od).")
        sys.exit(1)

    cfg.ensure_directories()
    print("=" * 60)
    print("          ROUTING101 - ADL DATASET EXTRACTION PIPELINE       ")
    print("=" * 60)
    print(f" Raw Data Dir      : {cfg.ADL_RAW_DIR}")
    print(f" Extracted Data Dir: {cfg.ADL_EXTRACTED_DIR}")
    print(f" Device            : {args.device}")
    print(f" Batch Size        : {args.batch_size}")
    print("=" * 60)

    if args.all or args.keyframes:
        print("\n>>> [STAGE 1/8] Extracting Keyframes & Map-Keyframes...")
        from .preprocess_keyframes import process_all_videos
        process_all_videos(sample_fps=args.sample_fps, overwrite=args.overwrite)

    if args.all or args.siglip:
        print("\n>>> [STAGE 2/8] Extracting SigLIP2 Frame Embeddings...")
        from .extract_siglip2 import run as run_siglip
        run_siglip(batch_size=args.batch_size, device=args.device, overwrite=args.overwrite)

    if args.all or args.clip:
        print("\n>>> [STAGE 3/8] Extracting CLIP ViT-B/32 Frame Embeddings...")
        from .extract_clip import run as run_clip
        run_clip(batch_size=args.batch_size, device=args.device, overwrite=args.overwrite)

    if args.all or args.caption:
        print("\n>>> [STAGE 4/8] Generating Dense Captions & SigLIP2 Caption Embeddings...")
        from .extract_caption import run as run_caption
        run_caption(model_type=args.caption_model, device=args.device, overwrite=args.overwrite)

    if args.all or args.ocr:
        print("\n>>> [STAGE 5/8] Extracting OCR Texts...")
        from .extract_ocr import run as run_ocr
        run_ocr(engine_name=args.ocr_engine, use_gpu=(args.device == "cuda"), overwrite=args.overwrite)

    if args.all or args.asr:
        print("\n>>> [STAGE 6/8] Extracting Speech Transcripts (Whisper ASR)...")
        from .extract_asr import run as run_asr
        run_asr(model_size=args.whisper_size, device=args.device, overwrite=args.overwrite)

    if args.all or args.summary:
        print("\n>>> [STAGE 7/8] Generating Video Summaries & Embeddings...")
        from .extract_summary import run as run_summary
        run_summary(device=args.device, overwrite=args.overwrite)

    if args.all or args.od:
        print("\n>>> [STAGE 8/8] Extracting Object Detection Labels & Building Vocabulary...")
        from .extract_od import run as run_od
        run_od(model_name=args.yolo_model, conf=args.od_conf, device=args.device, batch_size=args.batch_size, overwrite=args.overwrite)

    print("\n" + "=" * 60)
    print("          ALL ADL EXTRACTION STAGES COMPLETED!             ")
    print("=" * 60)


if __name__ == "__main__":
    main()
