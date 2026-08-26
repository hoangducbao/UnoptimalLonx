"""
pipeline/preprocess_keyframes.py -- Extract keyframes from ADL videos and generate map-keyframes CSVs.

Usage:
    python -m pipeline.preprocess_keyframes [--sample-fps 1.0] [--video-dir PATH]
"""

import argparse
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

from . import config_adl as cfg


def extract_keyframes_from_video(
    video_path: Path,
    output_keyframe_dir: Path,
    map_csv_path: Path,
    sample_fps: float = 1.0,
    min_scene_diff: float = 27.0,
    max_interval_sec: float = 2.0,
):
    """
    Extracts keyframes based on scene boundary detection + maximum time interval.
    Generates:
        1. JPEG keyframes: {output_keyframe_dir}/{n:03d}.jpg
        2. Map CSV: {map_csv_path} with columns: [n, pts_time, fps, frame_idx]
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or pd.isna(fps):
        fps = 30.0  # fallback default
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_keyframe_dir.mkdir(parents=True, exist_ok=True)
    map_csv_path.parent.mkdir(parents=True, exist_ok=True)

    prev_gray = None
    last_saved_time = -999.0
    last_saved_frame_idx = -1

    rows = []
    n = 1

    # Step size for reading frames to speed up processing
    frame_step = max(1, int(fps / sample_fps)) if sample_fps > 0 else 1

    current_frame_idx = 0
    while current_frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        pts_time = current_frame_idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (160, 90))

        should_save = False

        if prev_gray is None:
            # Always save first frame
            should_save = True
        else:
            time_diff = pts_time - last_saved_time
            if time_diff >= max_interval_sec:
                should_save = True
            else:
                # Calculate mean absolute difference
                diff = cv2.absdiff(gray_small, prev_gray)
                mean_diff = float(diff.mean())
                if mean_diff >= min_scene_diff:
                    should_save = True

        if should_save:
            frame_filename = output_keyframe_dir / f"{n:03d}.jpg"
            cv2.imwrite(str(frame_filename), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            rows.append({
                "n": n,
                "pts_time": round(pts_time, 3),
                "fps": round(fps, 2),
                "frame_idx": current_frame_idx
            })
            prev_gray = gray_small
            last_saved_time = pts_time
            last_saved_frame_idx = current_frame_idx
            n += 1

        current_frame_idx += frame_step

    cap.release()

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(map_csv_path, index=False)
        return True
    return False


def process_all_videos(sample_fps: float = 1.0, overwrite: bool = False):
    cfg.ensure_directories()
    video_files = sorted(list(cfg.VIDEO_DIR.glob("*.mp4")) + list(cfg.VIDEO_DIR.glob("*.mkv")) + list(cfg.VIDEO_DIR.glob("*.avi")))

    if not video_files:
        print(f"[Warning] No video files found in {cfg.VIDEO_DIR}")
        print(f"Please place ADL video files (.mp4) in: {cfg.VIDEO_DIR}")
        return

    print(f"[Preprocess] Found {len(video_files)} videos in {cfg.VIDEO_DIR}")
    for v_path in tqdm(video_files, desc="Extracting Keyframes"):
        video_id = v_path.stem
        out_dir = cfg.KEYFRAME_DIR / video_id
        map_path = cfg.MAP_KEYFRAMES_DIR / f"{video_id}.csv"

        if map_path.exists() and not overwrite:
            continue

        extract_keyframes_from_video(
            video_path=v_path,
            output_keyframe_dir=out_dir,
            map_csv_path=map_path,
            sample_fps=sample_fps
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keyframes and map-keyframes CSVs for ADL videos")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Sampling frequency (frames per second)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extractions")
    args = parser.parse_args()

    process_all_videos(sample_fps=args.sample_fps, overwrite=args.overwrite)
