"""
probe_external_oof_blend.py

Screen any leak-free OOF probability candidate (.npy, shape N x 14) against
Qwen or Qwen+tree baseline. Use this for prototype_oof, linear_oof, NB-SVM, etc.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import f1_score

from feature_utils_qwen_v4 import ALL_CLASSES, ACTION_TO_FAMILY

NUM_CLASSES = len(ALL_CLASSES)
LABEL2ID = {label: i for i, label in enumerate(ALL_CLASSES)}


def load_jsonl_ids(path: Path) -> List[str]:
    import json as _json
    ids = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(str(_json.loads(line)["id"]))
    return ids


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0))


def stable_softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_labels(data_path: Path, labels_csv: Path) -> np.ndarray:
    ids = load_jsonl_ids(data_path)
    with labels_csv.open(encoding="utf-8", newline="") as f:
        label_map = {str(row["id"]): str(row["action"]) for row in csv.DictReader(f)}
    return np.asarray([LABEL2ID[label_map[i]] for i in ids], dtype=np.int64)


def qwen_oof_proba(qwen_oof_path: Path, postprocess_path: Path, labels: np.ndarray) -> np.ndarray:
    oof = np.load(qwen_oof_path)
    action = oof["action_logits"].astype(np.float64)
    family = oof["family_logits"].astype(np.float64)
    oof_labels = oof["labels"].astype(np.int64)
    if not np.array_equal(labels, oof_labels):
        raise RuntimeError("Qwen OOF labels do not align with labels file.")

    post = json.loads(postprocess_path.read_text(encoding="utf-8"))
    family_index = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)
    class_weights = np.asarray(post["training_class_weights"], dtype=np.float64)
    final = (
        action / float(post["action_temperature"])
        + float(post["family_weight"]) * family[:, family_index]
        - float(post["prior_beta"]) * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )
    return stable_softmax(final)


def blend_logprob(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    logp = (1.0 - w) * np.log(np.maximum(a, 1e-12)) + w * np.log(np.maximum(b, 1e-12))
    return stable_softmax(logp)


def disagreement_report(base_proba: np.ndarray, cand_proba: np.ndarray, labels: np.ndarray) -> dict:
    base_pred = base_proba.argmax(axis=1)
    cand_pred = cand_proba.argmax(axis=1)
    base_correct = base_pred == labels
    cand_correct = cand_pred == labels
    wrong_correct = int(((~base_correct) & cand_correct).sum())
    correct_wrong = int((base_correct & (~cand_correct)).sum())
    oracle_pred = np.where(base_correct, base_pred, np.where(cand_correct, cand_pred, base_pred))
    return {
        "baseline_f1": macro_f1(labels, base_pred),
        "candidate_f1": macro_f1(labels, cand_pred),
        "baseline_wrong_candidate_correct": wrong_correct,
        "baseline_correct_candidate_wrong": correct_wrong,
        "complementarity_ratio": wrong_correct / max(1, correct_wrong),
        "oracle_f1": macro_f1(labels, oracle_pred),
        "oracle_gain": macro_f1(labels, oracle_pred) - macro_f1(labels, base_pred),
    }


def holdout_check(base_proba: np.ndarray, cand_proba: np.ndarray, labels: np.ndarray, w: float, n_seeds: int, frac: float) -> dict:
    rng_master = np.random.default_rng(0)
    gains = []
    n = len(labels)
    size = int(n * frac)
    for _ in range(n_seeds):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        idx = rng.choice(n, size=size, replace=False)
        base_f1 = macro_f1(labels[idx], base_proba[idx].argmax(axis=1))
        pred = blend_logprob(base_proba[idx], cand_proba[idx], w).argmax(axis=1)
        gains.append(macro_f1(labels[idx], pred) - base_f1)
    gains = np.asarray(gains, dtype=np.float64)
    return {
        "weight": float(w),
        "holdout_gain_mean": float(gains.mean()),
        "holdout_gain_min": float(gains.min()),
        "holdout_positive": int((gains > 0).sum()),
        "holdout_total": int(n_seeds),
        "gains": [float(x) for x in gains],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--qwen-oof", type=Path, required=True)
    p.add_argument("--postprocess", type=Path, required=True)
    p.add_argument("--candidate-oof-proba", type=Path, required=True)
    p.add_argument("--baseline-tree-oof-proba", type=Path, default=None)
    p.add_argument("--baseline-tree-weight", type=float, default=0.25)
    p.add_argument("--blend-weight", type=float, default=0.05)
    p.add_argument("--sweep-blend-weight", action="store_true")
    p.add_argument("--holdout-seeds", type=int, default=15)
    p.add_argument("--holdout-frac", type=float, default=0.5)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.data, args.labels_csv)
    qwen = qwen_oof_proba(args.qwen_oof, args.postprocess, labels)
    base = qwen
    base_name = "qwen"
    if args.baseline_tree_oof_proba is not None:
        tree = np.load(args.baseline_tree_oof_proba).astype(np.float64)
        if tree.shape != qwen.shape:
            raise RuntimeError(f"tree shape {tree.shape} != qwen shape {qwen.shape}")
        base = blend_logprob(qwen, tree, args.baseline_tree_weight)
        base_name = f"qwen_tree_w{args.baseline_tree_weight}"

    cand = np.load(args.candidate_oof_proba).astype(np.float64)
    if cand.shape != qwen.shape:
        raise RuntimeError(f"candidate shape {cand.shape} != qwen shape {qwen.shape}")
    cand = cand / np.maximum(cand.sum(axis=1, keepdims=True), 1e-12)

    print(f"Baseline: {base_name} Macro-F1={macro_f1(labels, base.argmax(axis=1)):.6f}")
    print(f"Candidate Macro-F1={macro_f1(labels, cand.argmax(axis=1)):.6f}")
    disagreement = disagreement_report(base, cand, labels)
    print("\n=== Disagreement / oracle ===")
    print(json.dumps(disagreement, indent=2))

    baseline_f1 = macro_f1(labels, base.argmax(axis=1))
    weights = [args.blend_weight]
    if args.sweep_blend_weight:
        weights = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]
    best = {"weight": args.blend_weight, "full_f1": -1.0, "gain": -999.0}
    print("\n=== Full blend sweep ===")
    for w in weights:
        pred = blend_logprob(base, cand, w).argmax(axis=1)
        f1 = macro_f1(labels, pred)
        gain = f1 - baseline_f1
        print(f"  w={w:.3f} f1={f1:.6f} gain={gain:+.6f}")
        if gain > best["gain"]:
            best = {"weight": float(w), "full_f1": float(f1), "gain": float(gain)}

    print("Best full-data weight:", best)
    holdout = holdout_check(base, cand, labels, best["weight"], args.holdout_seeds, args.holdout_frac)
    print("\n=== Holdout check ===")
    print(json.dumps(holdout, indent=2))

    if holdout["holdout_positive"] / holdout["holdout_total"] >= 0.8 and holdout["holdout_gain_mean"] > 0.0005:
        verdict = "PROMISING"
    elif best["gain"] > 0 and holdout["holdout_gain_mean"] > 0:
        verdict = "MARGINAL"
    else:
        verdict = "DISCARD"
    print("Verdict:", verdict)

    result = {"baseline": base_name, "disagreement": disagreement, "best_full": best, "holdout": holdout, "verdict": verdict}
    if args.report:
        (args.output_dir / "external_oof_blend_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
