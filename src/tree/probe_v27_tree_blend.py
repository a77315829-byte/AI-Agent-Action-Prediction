from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score

from feature_utils_qwen_v4 import ACTION_TO_FAMILY, ALL_CLASSES

NUM_CLASSES = len(ALL_CLASSES)
FAMILY_INDEX = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)


def final_logits_numpy(action_logits: np.ndarray, family_logits: np.ndarray, class_weights: np.ndarray, postprocess: Dict) -> np.ndarray:
    return (
        action_logits.astype(np.float64) / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"]) * family_logits.astype(np.float64)[:, FAMILY_INDEX]
        - float(postprocess["prior_beta"]) * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def macro_f1(labels: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(labels, pred, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0))


def blend_pred(qwen_prob: np.ndarray, tree_prob: np.ndarray, weight: float) -> np.ndarray:
    logp = (1.0 - weight) * np.log(np.maximum(qwen_prob, 1e-12)) + weight * np.log(np.maximum(tree_prob, 1e-12))
    return logp.argmax(axis=1).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-logits", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--tree-prob", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("model/v27_tree_blend_probe"))
    parser.add_argument("--max-weight", type=float, default=0.50)
    parser.add_argument("--weight-step", type=float, default=0.025)
    parser.add_argument("--holdout-repeats", type=int, default=10)
    parser.add_argument("--holdout-size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    z = np.load(args.qwen_logits)
    action_logits = z["action_logits"].astype(np.float64)
    family_logits = z["family_logits"].astype(np.float64)
    labels = z["labels"].astype(np.int64)
    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))
    class_weights = np.asarray(postprocess["training_class_weights"], dtype=np.float64)

    tree_prob = np.load(args.tree_prob).astype(np.float64)
    if tree_prob.shape != (len(labels), NUM_CLASSES):
        raise RuntimeError(f"tree_prob shape mismatch: {tree_prob.shape} != {(len(labels), NUM_CLASSES)}")
    tree_prob /= np.maximum(tree_prob.sum(axis=1, keepdims=True), 1e-12)

    final_logits = final_logits_numpy(action_logits, family_logits, class_weights, postprocess)
    qwen_prob = softmax(final_logits)
    qwen_pred = qwen_prob.argmax(axis=1)
    qwen_score = macro_f1(labels, qwen_pred)
    tree_score = macro_f1(labels, tree_prob.argmax(axis=1))
    fixed025 = macro_f1(labels, blend_pred(qwen_prob, tree_prob, 0.25))

    weights = np.round(np.arange(0.0, args.max_weight + 1e-9, args.weight_step), 6)
    candidates: List[dict] = []
    for w in weights:
        score = macro_f1(labels, blend_pred(qwen_prob, tree_prob, float(w)))
        candidates.append({"weight": float(w), "score": float(score), "gain_vs_qwen": float(score - qwen_score)})
    best = max(candidates, key=lambda x: x["score"])

    rng = np.random.default_rng(args.seed)
    gains = []
    gains_vs_fixed = []
    holdouts = []
    n = len(labels)
    holdout_n = int(round(n * args.holdout_size)) if args.holdout_size < 1 else int(args.holdout_size)
    holdout_n = max(1, min(n, holdout_n))
    for i in range(args.holdout_repeats):
        idx = rng.choice(n, size=holdout_n, replace=False)
        base = macro_f1(labels[idx], qwen_pred[idx])
        fixed = macro_f1(labels[idx], blend_pred(qwen_prob[idx], tree_prob[idx], 0.25))
        local_best = None
        for w in weights:
            score = macro_f1(labels[idx], blend_pred(qwen_prob[idx], tree_prob[idx], float(w)))
            if local_best is None or score > local_best["score"]:
                local_best = {"weight": float(w), "score": float(score)}
        gain = float(local_best["score"] - base)
        gain_fixed = float(local_best["score"] - fixed)
        gains.append(gain)
        gains_vs_fixed.append(gain_fixed)
        holdouts.append({"repeat": i + 1, "gain_vs_qwen": gain, "gain_vs_fixed025": gain_fixed, **local_best})

    summary = {
        "qwen_macro_f1": float(qwen_score),
        "tree_macro_f1": float(tree_score),
        "fixed_w025_macro_f1": float(fixed025),
        "fixed_w025_gain_vs_qwen": float(fixed025 - qwen_score),
        "best_full": best,
        "holdout_gain_mean_vs_qwen": float(np.mean(gains)),
        "holdout_gain_min_vs_qwen": float(np.min(gains)),
        "holdout_positive_vs_qwen": int(np.sum(np.asarray(gains) > 0)),
        "holdout_gain_mean_vs_fixed025": float(np.mean(gains_vs_fixed)),
        "holdout_gain_min_vs_fixed025": float(np.min(gains_vs_fixed)),
        "holdout_positive_vs_fixed025": int(np.sum(np.asarray(gains_vs_fixed) > 0)),
        "holdout_repeats": holdouts,
        "top_candidates": sorted(candidates, key=lambda x: x["score"], reverse=True)[:20],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.report else json.dumps({k: summary[k] for k in summary if k != "holdout_repeats" and k != "top_candidates"}, ensure_ascii=False, indent=2))
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
