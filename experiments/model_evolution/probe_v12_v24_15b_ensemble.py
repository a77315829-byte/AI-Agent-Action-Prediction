#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Probe Qwen 0.5B V12 + Qwen 1.5B V24 ensemble on the same validation fold.

Example:
python .\probe_v12_v24_15b_ensemble.py `
  --v12-logits .\model\qwen_distill_v12_eval\validation_logits_v12.npz `
  --v15-logits .\model\qwen_segment_v24_15b_r16_eval\validation_logits_v24_15b.npz `
  --v15-postprocess .\model\qwen_segment_v24_15b_r16_eval\postprocess.json `
  --output-dir .\model\v12_v24_15b_ensemble_probe

V12 defaults are embedded from the known best configuration:
- action_temperature=0.6
- family_weight=0.15
- prior_beta=0.25
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]

ACTION_TO_FAMILY = np.asarray(
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 3],
    dtype=np.int64,
)

# Training class weights used by the V12 postprocess.
V12_TRAINING_CLASS_WEIGHTS = np.asarray(
    [
        0.8572690251352013,
        0.84274223991845,
        1.0366924510397455,
        0.9862908339428644,
        0.8179509647354836,
        1.3554499048325421,
        1.009072807992434,
        0.996587290637956,
        1.0232914644762283,
        1.2165721371457607,
        1.166404985555288,
        1.1687102789899324,
        1.407913957899642,
        0.9913123924717526,
    ],
    dtype=np.float64,
)

GROUPS = {
    "file_search": [0, 1, 2, 3],
    "edit_write_patch": [4, 5, 6],
    "exec_check": [7, 8, 9],
    "ask_plan": [10, 11],
    "web_search": [12],
    "respond_only": [13],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v12-logits", type=Path, required=True)
    parser.add_argument("--v15-logits", type=Path, required=True)
    parser.add_argument("--v15-postprocess", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--v12-action-temperature", type=float, default=0.6)
    parser.add_argument("--v12-family-weight", type=float, default=0.15)
    parser.add_argument("--v12-prior-beta", type=float, default=0.25)

    parser.add_argument(
        "--weights",
        type=str,
        default="0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6",
        help="Comma-separated 1.5B blend weights.",
    )
    parser.add_argument("--holdout-repeats", type=int, default=50)
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.maximum(exp_x.sum(axis=1, keepdims=True), 1e-300)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def validate_logits(name: str, payload: Dict[str, np.ndarray]) -> None:
    required = {"action_logits", "family_logits", "labels"}
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"{name}: missing keys {sorted(missing)}")

    n = payload["labels"].shape[0]
    if payload["action_logits"].shape != (n, len(CLASSES)):
        raise RuntimeError(
            f"{name}: action_logits shape={payload['action_logits'].shape}, "
            f"expected ({n}, {len(CLASSES)})"
        )
    if payload["family_logits"].shape[0] != n:
        raise RuntimeError(f"{name}: family_logits rows do not match labels")


def compute_probability(
    payload: Dict[str, np.ndarray],
    action_temperature: float,
    family_weight: float,
    prior_beta: float,
    training_class_weights: np.ndarray | None,
) -> np.ndarray:
    if action_temperature <= 0:
        raise ValueError("action_temperature must be positive")

    final_logits = (
        payload["action_logits"].astype(np.float64)
        / float(action_temperature)
    )
    if family_weight != 0:
        final_logits = (
            final_logits
            + float(family_weight)
            * payload["family_logits"].astype(np.float64)[:, ACTION_TO_FAMILY]
        )

    if prior_beta != 0:
        if training_class_weights is None:
            raise RuntimeError(
                "prior_beta is non-zero but training_class_weights are missing"
            )
        weights = np.asarray(training_class_weights, dtype=np.float64)
        if weights.shape != (len(CLASSES),):
            raise RuntimeError(
                f"training_class_weights shape={weights.shape}, expected {(len(CLASSES),)}"
            )
        final_logits = (
            final_logits
            - float(prior_beta)
            * np.log(np.maximum(weights, 1e-12))[None, :]
        )

    return stable_softmax(final_logits)


def load_v15_postprocess(path: Path | None) -> dict:
    if path is None:
        return {
            "action_temperature": 1.0,
            "family_weight": 0.2,
            "prior_beta": 0.0,
            "training_class_weights": None,
        }

    post = json.loads(path.read_text(encoding="utf-8"))
    return {
        "action_temperature": float(post.get("action_temperature", 1.0)),
        "family_weight": float(post.get("family_weight", 0.0)),
        "prior_beta": float(post.get("prior_beta", 0.0)),
        "training_class_weights": post.get("training_class_weights"),
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(len(CLASSES))),
            average="macro",
            zero_division=0,
        )
    )


def blend_log_prob(
    v12_probability: np.ndarray,
    v15_probability: np.ndarray,
    v15_weight: float,
) -> np.ndarray:
    weight = float(v15_weight)
    log_probability = (
        (1.0 - weight)
        * np.log(np.maximum(v12_probability, 1e-12))
        + weight
        * np.log(np.maximum(v15_probability, 1e-12))
    )
    return stable_softmax(log_probability)


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class_id": np.arange(len(CLASSES)),
            "class": CLASSES,
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def topk_accuracy(probability: np.ndarray, labels: np.ndarray, k: int) -> float:
    topk = np.argpartition(probability, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(topk == labels[:, None], axis=1)))


def oracle_predictions(
    labels: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> np.ndarray:
    # Optimistic upper bound: select the correct model whenever either is correct.
    result = pred_a.copy()
    use_b = (pred_a != labels) & (pred_b == labels)
    result[use_b] = pred_b[use_b]
    return result


def group_gated_prediction(
    base_probability: np.ndarray,
    specialist_probability: np.ndarray,
    group_ids: Iterable[int],
    specialist_weight: float,
) -> np.ndarray:
    ids = np.asarray(list(group_ids), dtype=np.int64)
    base_pred = base_probability.argmax(axis=1)
    specialist_pred = specialist_probability.argmax(axis=1)

    # Only alter rows where both models' argmax predictions are inside the same
    # confusion group. This avoids letting a weak specialist affect unrelated rows.
    eligible = np.isin(base_pred, ids) & np.isin(specialist_pred, ids)

    blended = blend_log_prob(
        base_probability,
        specialist_probability,
        specialist_weight,
    )
    result = base_pred.copy()
    result[eligible] = blended.argmax(axis=1)[eligible]
    return result


def stratified_random_holdouts(
    labels: np.ndarray,
    repeats: int,
    fraction: float,
    seed: int,
) -> List[np.ndarray]:
    if not 0 < fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    by_class = {
        class_id: np.flatnonzero(labels == class_id)
        for class_id in range(len(CLASSES))
    }
    holdouts: List[np.ndarray] = []

    for _ in range(repeats):
        selected: List[np.ndarray] = []
        for indices in by_class.values():
            count = max(1, int(round(len(indices) * fraction)))
            selected.append(rng.choice(indices, size=count, replace=False))
        holdouts.append(np.sort(np.concatenate(selected)))

    return holdouts


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v12 = load_npz(args.v12_logits)
    v15 = load_npz(args.v15_logits)
    validate_logits("V12", v12)
    validate_logits("V15", v15)

    y12 = v12["labels"].astype(np.int64)
    y15 = v15["labels"].astype(np.int64)

    if not np.array_equal(y12, y15):
        raise RuntimeError("V12 and V15 label arrays are not identical.")

    if "validation_indices" in v15 and "validation_indices" in v12:
        if not np.array_equal(v12["validation_indices"], v15["validation_indices"]):
            raise RuntimeError("V12 and V15 validation_indices are not identical.")

    y = y12

    v12_probability = compute_probability(
        v12,
        action_temperature=args.v12_action_temperature,
        family_weight=args.v12_family_weight,
        prior_beta=args.v12_prior_beta,
        training_class_weights=V12_TRAINING_CLASS_WEIGHTS,
    )

    v15_post = load_v15_postprocess(args.v15_postprocess)
    v15_probability = compute_probability(
        v15,
        action_temperature=v15_post["action_temperature"],
        family_weight=v15_post["family_weight"],
        prior_beta=v15_post["prior_beta"],
        training_class_weights=(
            None
            if v15_post["training_class_weights"] is None
            else np.asarray(v15_post["training_class_weights"], dtype=np.float64)
        ),
    )

    pred12 = v12_probability.argmax(axis=1)
    pred15 = v15_probability.argmax(axis=1)

    score12 = macro_f1(y, pred12)
    score15 = macro_f1(y, pred15)
    disagreement = float(np.mean(pred12 != pred15))
    v12_wrong_v15_right = int(np.sum((pred12 != y) & (pred15 == y)))
    v15_wrong_v12_right = int(np.sum((pred15 != y) & (pred12 == y)))

    oracle_pred = oracle_predictions(y, pred12, pred15)
    oracle_score = macro_f1(y, oracle_pred)

    print("=== Alignment ===")
    print("Rows:", len(y))
    print("Labels identical: yes")
    if "validation_indices" in v15:
        print("V15 validation_indices:", v15["validation_indices"].shape)

    print("\n=== Standalone ===")
    print(f"V12 Macro-F1: {score12:.9f}")
    print(f"V15 Macro-F1: {score15:.9f}")
    print(f"Prediction disagreement: {disagreement:.6%}")
    print(f"V12 wrong / V15 right: {v12_wrong_v15_right}")
    print(f"V15 wrong / V12 right: {v15_wrong_v12_right}")
    print(f"Oracle selector Macro-F1: {oracle_score:.9f}")
    print(f"Oracle gain over V12: {oracle_score - score12:+.9f}")

    weights = sorted(
        {
            float(item.strip())
            for item in args.weights.split(",")
            if item.strip()
        }
    )

    blend_rows = []
    blended_predictions: Dict[float, np.ndarray] = {}
    for weight in weights:
        probability = blend_log_prob(v12_probability, v15_probability, weight)
        prediction = probability.argmax(axis=1)
        blended_predictions[weight] = prediction
        blend_rows.append(
            {
                "v15_weight": weight,
                "macro_f1": macro_f1(y, prediction),
                "gain_vs_v12": macro_f1(y, prediction) - score12,
                "accuracy": float(accuracy_score(y, prediction)),
                "changed_vs_v12": int(np.sum(prediction != pred12)),
                "top2_accuracy": topk_accuracy(probability, y, 2),
                "top3_accuracy": topk_accuracy(probability, y, 3),
            }
        )

    blend_df = pd.DataFrame(blend_rows).sort_values(
        ["macro_f1", "v15_weight"],
        ascending=[False, True],
    )
    best_weight = float(blend_df.iloc[0]["v15_weight"])
    best_score = float(blend_df.iloc[0]["macro_f1"])
    best_pred = blended_predictions[best_weight]

    print("\n=== Global blend sweep ===")
    print(blend_df.sort_values("v15_weight").to_string(index=False))
    print(
        f"\nBest global blend: v15_weight={best_weight:.3f}, "
        f"Macro-F1={best_score:.9f}, gain={best_score - score12:+.9f}"
    )

    gated_rows = []
    for group_name, ids in GROUPS.items():
        for weight in weights:
            prediction = group_gated_prediction(
                v12_probability,
                v15_probability,
                ids,
                weight,
            )
            gated_rows.append(
                {
                    "group": group_name,
                    "v15_weight": weight,
                    "macro_f1": macro_f1(y, prediction),
                    "gain_vs_v12": macro_f1(y, prediction) - score12,
                    "changed_vs_v12": int(np.sum(prediction != pred12)),
                }
            )

    gated_df = pd.DataFrame(gated_rows)
    best_gated = gated_df.sort_values(
        ["macro_f1", "changed_vs_v12"],
        ascending=[False, True],
    ).iloc[0]

    print("\n=== Best gated result per group ===")
    print(
        gated_df.sort_values(
            ["group", "macro_f1"],
            ascending=[True, False],
        )
        .groupby("group", as_index=False)
        .head(1)
        .sort_values("macro_f1", ascending=False)
        .to_string(index=False)
    )

    holdouts = stratified_random_holdouts(
        y,
        repeats=args.holdout_repeats,
        fraction=args.holdout_fraction,
        seed=args.seed,
    )

    holdout_rows = []
    candidate_weights = sorted(
        set(
            blend_df.head(min(5, len(blend_df)))["v15_weight"]
            .astype(float)
            .tolist()
        )
    )

    for weight in candidate_weights:
        prediction = blended_predictions[weight]
        gains = []
        for repeat_id, indices in enumerate(holdouts):
            base_score = macro_f1(y[indices], pred12[indices])
            candidate_score = macro_f1(y[indices], prediction[indices])
            gain = candidate_score - base_score
            gains.append(gain)
            holdout_rows.append(
                {
                    "mode": "global",
                    "name": f"w={weight}",
                    "repeat": repeat_id,
                    "gain": gain,
                }
            )

        print(
            f"Holdout global w={weight:.3f}: "
            f"mean={np.mean(gains):+.9f}, "
            f"min={np.min(gains):+.9f}, "
            f"max={np.max(gains):+.9f}, "
            f"positive={np.sum(np.asarray(gains) > 0)}/{len(gains)}"
        )

    best_gated_group = str(best_gated["group"])
    best_gated_weight = float(best_gated["v15_weight"])
    best_gated_pred = group_gated_prediction(
        v12_probability,
        v15_probability,
        GROUPS[best_gated_group],
        best_gated_weight,
    )
    gated_gains = []
    for repeat_id, indices in enumerate(holdouts):
        base_score = macro_f1(y[indices], pred12[indices])
        candidate_score = macro_f1(y[indices], best_gated_pred[indices])
        gain = candidate_score - base_score
        gated_gains.append(gain)
        holdout_rows.append(
            {
                "mode": "gated",
                "name": f"{best_gated_group}_w={best_gated_weight}",
                "repeat": repeat_id,
                "gain": gain,
            }
        )

    print(
        f"Holdout gated {best_gated_group} w={best_gated_weight:.3f}: "
        f"mean={np.mean(gated_gains):+.9f}, "
        f"min={np.min(gated_gains):+.9f}, "
        f"max={np.max(gated_gains):+.9f}, "
        f"positive={np.sum(np.asarray(gated_gains) > 0)}/{len(gated_gains)}"
    )

    class_v12 = class_metrics(y, pred12).rename(
        columns={
            "precision": "v12_precision",
            "recall": "v12_recall",
            "f1": "v12_f1",
        }
    )
    class_v15 = class_metrics(y, pred15)[
        ["class_id", "precision", "recall", "f1"]
    ].rename(
        columns={
            "precision": "v15_precision",
            "recall": "v15_recall",
            "f1": "v15_f1",
        }
    )
    class_best = class_metrics(y, best_pred)[
        ["class_id", "precision", "recall", "f1"]
    ].rename(
        columns={
            "precision": "blend_precision",
            "recall": "blend_recall",
            "f1": "blend_f1",
        }
    )
    class_df = (
        class_v12.merge(class_v15, on="class_id")
        .merge(class_best, on="class_id")
    )
    class_df["v15_gain_vs_v12"] = class_df["v15_f1"] - class_df["v12_f1"]
    class_df["blend_gain_vs_v12"] = (
        class_df["blend_f1"] - class_df["v12_f1"]
    )

    cm12 = pd.DataFrame(
        confusion_matrix(y, pred12, labels=list(range(len(CLASSES)))),
        index=[f"true_{name}" for name in CLASSES],
        columns=[f"pred_{name}" for name in CLASSES],
    )
    cm15 = pd.DataFrame(
        confusion_matrix(y, pred15, labels=list(range(len(CLASSES)))),
        index=[f"true_{name}" for name in CLASSES],
        columns=[f"pred_{name}" for name in CLASSES],
    )

    summary = {
        "rows": int(len(y)),
        "v12_macro_f1": score12,
        "v15_macro_f1": score15,
        "prediction_disagreement_rate": disagreement,
        "v12_wrong_v15_right": v12_wrong_v15_right,
        "v15_wrong_v12_right": v15_wrong_v12_right,
        "oracle_macro_f1": oracle_score,
        "oracle_gain_vs_v12": oracle_score - score12,
        "best_global_weight": best_weight,
        "best_global_macro_f1": best_score,
        "best_global_gain_vs_v12": best_score - score12,
        "best_gated_group": best_gated_group,
        "best_gated_weight": best_gated_weight,
        "best_gated_macro_f1": float(best_gated["macro_f1"]),
        "best_gated_gain_vs_v12": float(best_gated["gain_vs_v12"]),
        "v15_postprocess": v15_post,
    }

    blend_df.sort_values("v15_weight").to_csv(
        args.output_dir / "global_blend_sweep.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gated_df.to_csv(
        args.output_dir / "gated_blend_sweep.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(holdout_rows).to_csv(
        args.output_dir / "holdout_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    class_df.to_csv(
        args.output_dir / "class_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cm12.to_csv(
        args.output_dir / "confusion_matrix_v12.csv",
        encoding="utf-8-sig",
    )
    cm15.to_csv(
        args.output_dir / "confusion_matrix_v15.csv",
        encoding="utf-8-sig",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Final summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
