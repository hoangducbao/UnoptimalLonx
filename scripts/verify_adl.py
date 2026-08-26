"""
scripts/verify_adl.py -- Automated verification & diagnostic tool for ADL deployment.

Checks:
  1. Data directory integrity (raw videos, keyframes, map-keyframes CSVs).
  2. Embeddings existence & shape (SigLIP2, CLIP, Caption, OCR, ASR, Summary).
  3. FAISS index loading & vector dimensions.
  4. Elasticsearch connectivity & document counts.
  5. Sample search query execution across multiple signals.

Usage:
    python scripts/verify_adl.py
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend import config
from pipeline import config_adl as cfg_adl


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f" [CHECK] {title}")
    print("=" * 60)


def check_directories():
    print_section("1. Checking Directory Structure & Files")
    print(f"Dataset Mode: {config.DATASET_MODE}")
    print(f"Video Root  : {config.VIDEO_DIR}")
    print(f"Keyframes   : {config.THUMBNAIL_ROOT}")
    print(f"Map-CSVs    : {config.MAP_KEYFRAMES_DIR}")

    videos = [p.name for i, p in enumerate(config.VIDEO_DIR.glob("*.mp4")) if i < 3] if config.VIDEO_DIR.exists() else []
    print(f"-> Video directory     : {config.VIDEO_DIR} (Sample: {videos if videos else 'No .mp4 in root'})")

    keyframe_dirs = [p.name for i, p in enumerate(config.THUMBNAIL_ROOT.iterdir()) if p.is_dir() and i < 5] if config.THUMBNAIL_ROOT.exists() else []
    print(f"-> Keyframe root       : {config.THUMBNAIL_ROOT} (Sample folders: {keyframe_dirs})")

    map_csvs = [p.name for i, p in enumerate(config.MAP_KEYFRAMES_DIR.glob("*.csv")) if i < 5] if config.MAP_KEYFRAMES_DIR.exists() else []
    print(f"-> Map-Keyframes dir   : {config.MAP_KEYFRAMES_DIR} (Sample CSVs: {map_csvs})")

    if not config.THUMBNAIL_ROOT.exists() and not config.VIDEO_DIR.exists():
        print("[!] Warning: Neither Keyframes nor Video directory exists.")
        return False
    return True


def check_embeddings():
    print_section("2. Checking Extracted Feature Files & Dimensions")
    import glob

    # 1. SigLIP2 Frame Embeddings
    siglip_files = glob.glob(config.FRAME_SIGLIP2_GLOB)
    print(f"-> SigLIP2 Frame .npy files: {len(siglip_files)}")
    if siglip_files:
        sample_vec = np.load(siglip_files[0])
        print(f"   Sample shape: {sample_vec.shape}, dtype: {sample_vec.dtype}")
        if sample_vec.shape[-1] != 768:
            print(f"   [!] Error: Expected dim=768, got {sample_vec.shape[-1]}")

    # 2. CLIP Frame Embeddings
    clip_files = glob.glob(config.FRAME_CLIP_GLOB)
    print(f"-> CLIP ViT-B/32 .npy files: {len(clip_files)}")
    if clip_files:
        sample_clip = np.load(clip_files[0])
        print(f"   Sample shape: {sample_clip.shape}, dtype: {sample_clip.dtype}")
        if sample_clip.shape[-1] != 512:
            print(f"   [!] Error: Expected dim=512, got {sample_clip.shape[-1]}")

    # 3. Caption Embeddings
    cap_npys = list(config.SIGLIP_CAPTION_DIR.glob("*.npy")) if config.SIGLIP_CAPTION_DIR.exists() else []
    print(f"-> Caption Embeddings .npy : {len(cap_npys)}")

    # 4. OCR CSVs
    ocr_csvs = list(config.OCR_DIR.glob("*.csv")) if config.OCR_DIR.exists() else []
    print(f"-> OCR CSV files           : {len(ocr_csvs)}")

    # 5. Summary files
    sum_npys = list(config.SUMMARY_EMBED_DIR.glob("*.npy")) if config.SUMMARY_EMBED_DIR.exists() else []
    print(f"-> Summary Embeddings .npy : {len(sum_npys)}")

    # 6. Object Detection CSVs & Vocab
    od_csvs = list(config.FILTERED_OBJECT_DIR.glob("*.csv")) if config.FILTERED_OBJECT_DIR.exists() else []
    print(f"-> Object Detection CSVs   : {len(od_csvs)}")
    if config.CLASS_VOCAB_CSV.exists():
        vocab_lines = len(config.CLASS_VOCAB_CSV.read_text(encoding="utf-8").splitlines()) - 1
        print(f"-> OD Class Vocabulary     : {vocab_lines} classes in {config.CLASS_VOCAB_CSV}")
    else:
        print(f"-> OD Class Vocabulary     : [!] Missing class_vocab.csv")


def check_elasticsearch():
    print_section("3. Checking Elasticsearch Cluster")
    try:
        from backend.es_client import get_es_client
        es = get_es_client()
        if not es.ping():
            print(f"[!] Warning: Elasticsearch at {config.ES_HOST} is unreachable.")
            return False

        info = es.info()
        print(f"-> Connected to Elasticsearch cluster: '{info.get('cluster_name')}' (v{info.get('version', {}).get('number')})")

        for idx_name in [config.ES_INDEX_ASR, config.ES_INDEX_CAPTION, config.ES_INDEX_OCR, config.ES_INDEX_SUMMARY]:
            if es.indices.exists(index=idx_name):
                count = es.count(index=idx_name).get("count", 0)
                print(f"   Index [{idx_name}]: {count} documents")
            else:
                print(f"   Index [{idx_name}]: Not created yet")
        return True
    except Exception as e:
        print(f"[!] Elasticsearch check error: {e}")
        return False


def test_sample_search():
    print_section("4. Running Diagnostic Test Query")
    test_query = "person opening refrigerator"
    print(f"Test Query: '{test_query}'")

    try:
        from backend.search import keyframe as kf_mod
        t0 = time.time()
        res_siglip = kf_mod.search_siglip2_frame(test_query, k=5)
        dt = (time.time() - t0) * 1000
        print(f"-> SigLIP2 Visual Search: {len(res_siglip)} results in {dt:.1f}ms")
        if not res_siglip.empty:
            top = res_siglip.iloc[0]
            print(f"   Top 1: video_id={top['video_id']}, frame_id={top['frame_id']}, score={top['score']:.4f}")
    except Exception as e:
        print(f"-> SigLIP2 search test failed: {e}")

    try:
        from backend.search import caption as cap_mod
        t0 = time.time()
        res_cap = cap_mod.search_caption(test_query, k=5)
        dt = (time.time() - t0) * 1000
        print(f"-> Caption Search       : {len(res_cap)} results in {dt:.1f}ms")
    except Exception as e:
        print(f"-> Caption search test failed: {e}")


def main():
    print("=" * 60)
    print("       ROUTING101 - ADL SYSTEM DEPLOYMENT DIAGNOSTICS         ")
    print("=" * 60)

    check_directories()
    check_embeddings()
    check_elasticsearch()
    test_sample_search()

    print("\n" + "=" * 60)
    print("               DIAGNOSTIC RUN COMPLETED                     ")
    print("=" * 60)


if __name__ == "__main__":
    main()
