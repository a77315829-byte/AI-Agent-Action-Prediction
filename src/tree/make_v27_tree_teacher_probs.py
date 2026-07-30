from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.metrics import f1_score


def load_jsonl(path: Path) -> List[dict]:
    samples: List[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("v23_tree_submission_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_labels(path: Path, samples: List[dict], all_classes: List[str]) -> np.ndarray:
    label2id = {label: index for index, label in enumerate(all_classes)}
    with path.open(encoding="utf-8", newline="") as file:
        label_map = {str(row["id"]): label2id[str(row["action"])] for row in csv.DictReader(file)}
    return np.asarray([label_map[str(sample["id"])] for sample in samples], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create V27 tree-teacher probabilities for train.jsonl by reusing the "
            "V23 LightGBM tree artifact and V23 feature pipeline. This is a fast "
            "teacher-generation path; for strict local model selection, validate the "
            "student on held-out labels and keep V23 as the deployment baseline."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--tree-artifacts", type=Path, default=Path("model/v23_tree_lgbm_full/tree_artifacts.joblib"))
    parser.add_argument("--tree-script", type=Path, default=Path("script_submission_v23_qwen_tree_blend.py"))
    parser.add_argument("--output", type=Path, default=Path("model/v27_tree_teacher/tree_prob_all.npy"))
    parser.add_argument("--metadata-output", type=Path, default=Path("model/v27_tree_teacher/metadata.json"))
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    print("Load V23 tree feature/predict module:", args.tree_script)
    v23 = load_module(args.tree_script)

    print("Load data:", args.data)
    samples = load_jsonl(args.data)
    print("Rows:", len(samples))

    print("Load tree artifacts:", args.tree_artifacts)
    tree_bundle = joblib.load(args.tree_artifacts)

    parts: List[np.ndarray] = []
    for start in range(0, len(samples), args.chunk_size):
        end = min(start + args.chunk_size, len(samples))
        print(f"Predict tree probabilities: {start}:{end}")
        prob = v23.predict_tree_probabilities(tree_bundle, samples[start:end])
        prob = np.asarray(prob, dtype=np.float32)
        if prob.ndim != 2 or prob.shape[1] != len(v23.ALL_CLASSES):
            raise RuntimeError(f"Bad probability shape for chunk {start}:{end}: {prob.shape}")
        prob /= np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)
        parts.append(prob)

    all_prob = np.concatenate(parts, axis=0).astype(np.float32)
    if all_prob.shape != (len(samples), len(v23.ALL_CLASSES)):
        raise RuntimeError(f"Bad final probability shape: {all_prob.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, all_prob)
    print("Saved:", args.output, all_prob.shape)

    metadata = {
        "format": "v27_tree_teacher_probabilities",
        "rows": int(all_prob.shape[0]),
        "classes": list(v23.ALL_CLASSES),
        "tree_artifacts": str(args.tree_artifacts),
        "tree_script": str(args.tree_script),
        "chunk_size": int(args.chunk_size),
        "note": "Probabilities are generated from the supplied V23 tree artifact. If the artifact was trained on all train rows, use this as a fast teacher experiment, not as a leak-free standalone validation score.",
    }

    if args.labels_csv.exists():
        labels = load_labels(args.labels_csv, samples, list(v23.ALL_CLASSES))
        pred = all_prob.argmax(axis=1)
        macro = float(f1_score(labels, pred, labels=np.arange(len(v23.ALL_CLASSES)), average="macro", zero_division=0))
        metadata["tree_macro_f1_on_supplied_labels"] = macro
        if args.report:
            print(f"Tree Macro-F1 on supplied labels: {macro:.6f}")

    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved metadata:", args.metadata_output)


if __name__ == "__main__":
    main()
