"""
build_transition_matrix.py

Builds the DEPLOYMENT transition matrix from ALL 70,000 train rows' true
labels (no fold-splitting needed here -- that was only for leak-free
screening; for the artifact that ships in the submission zip, using the
full dataset is correct and standard).

Output: model/transition_matrix.npy (14, 14) row-normalized P(next|prev)
        model/viterbi_config.json  ({"lambda": ..., "smoothing": ...})
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils_qwen_v4 import ALL_CLASSES

NUM_CLASSES = len(ALL_CLASSES)
LABEL2ID = {label: i for i, label in enumerate(ALL_CLASSES)}


def load_jsonl(path: Path) -> List[dict]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def parse_session_step(sample_id: str) -> Tuple[str, int]:
    session, _, step_part = sample_id.rpartition("-step_")
    try:
        step = int(step_part)
    except ValueError:
        step = 0
    return session, step


def build_sessions(ids: List[str]) -> Dict[str, List[int]]:
    parsed = [(parse_session_step(sid), i) for i, sid in enumerate(ids)]
    sessions: Dict[str, List[Tuple[int, int]]] = {}
    for (session, step), idx in parsed:
        sessions.setdefault(session, []).append((step, idx))
    return {session: [idx for _, idx in sorted(steps)] for session, steps in sessions.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--smoothing", type=float, default=2.0)
    parser.add_argument("--lam", type=float, default=0.6)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(args.data)
    ids = [str(s["id"]) for s in samples]
    with args.labels_csv.open(encoding="utf-8", newline="") as f:
        label_map = {str(row["id"]): str(row["action"]) for row in csv.DictReader(f)}
    labels = np.asarray([LABEL2ID[label_map[i]] for i in ids], dtype=np.int64)

    sessions = build_sessions(ids)
    counts = np.full((NUM_CLASSES, NUM_CLASSES), args.smoothing, dtype=np.float64)
    for row_indices in sessions.values():
        for prev_idx, curr_idx in zip(row_indices[:-1], row_indices[1:]):
            counts[labels[prev_idx], labels[curr_idx]] += 1.0
    transition_matrix = counts / counts.sum(axis=1, keepdims=True)

    np.save(args.output_dir / "transition_matrix.npy", transition_matrix.astype(np.float64))
    (args.output_dir / "viterbi_config.json").write_text(
        json.dumps({"lambda": args.lam, "smoothing": args.smoothing, "classes": ALL_CLASSES}, indent=2),
        encoding="utf-8",
    )
    print(f"Sessions: {len(sessions)}, rows: {len(labels)}")
    print(f"Saved: {args.output_dir / 'transition_matrix.npy'}")
    print(f"Saved: {args.output_dir / 'viterbi_config.json'}")
    print(f"Config: lambda={args.lam}, smoothing={args.smoothing}")


if __name__ == "__main__":
    main()
