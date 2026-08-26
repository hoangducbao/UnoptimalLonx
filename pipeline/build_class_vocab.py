"""
pipeline/build_class_vocab.py -- offline, run-once preprocessing for the OD
(object-detection) text filter (see backend/od_filter.py). Scans every
per-video OD-detections CSV under AICDataExtracted/filtered_object/*.csv
(produced upstream by AICPreprocess/filter_apply.py, outside this repo --
one row per surviving detection, `class_name` is the Open Images label),
collects the unique class names, normalizes them, and writes the result as
a flat list to AICDataExtracted/filtered_object/class_vocab.csv.

Normalization is intentionally light -- lowercase + collapsed whitespace
only, no plural-stripping. Open Images class names include plural-looking
entries that are their own distinct classes (e.g. "Glasses" is not the
plural of some "Glass" class), so blindly stripping a trailing "s" risks
merging classes that aren't actually the same thing. Runtime matching
(backend/od_filter.py) uses fuzzy string matching, which already tolerates
a query typed as "cars" against the vocab entry "car" without this needing
to normalize plurals away up front.

Run once:
    python pipeline/build_class_vocab.py
Re-run only if the filtered_object/*.csv source data changes -- it
overwrites class_vocab.csv each time (same idempotent-rebuild spirit as
the FAISS/ES index builders in backend/, just without the exists()-guard
since this is a manual, occasional step rather than an eager startup one).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config  # noqa: E402


def normalize(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def main():
    src_dir = config.FILTERED_OBJECT_DIR
    csv_paths = sorted(p for p in src_dir.glob("*.csv") if p.name != "class_vocab.csv")
    if not csv_paths:
        raise FileNotFoundError(f"no per-video OD CSVs found under {src_dir}")

    names = set()
    for i, path in enumerate(csv_paths, start=1):
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "class_name" not in reader.fieldnames:
                continue
            for row in reader:
                raw = row.get("class_name")
                if raw:
                    names.add(normalize(raw))
        if i % 100 == 0:
            print(f"[{i}/{len(csv_paths)} videos] {len(names)} unique classes so far", flush=True)

    out_path = config.CLASS_VOCAB_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name"])
        for name in sorted(names):
            writer.writerow([name])

    print(f"\nDone. {len(csv_paths)} videos scanned, {len(names)} unique classes.")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
