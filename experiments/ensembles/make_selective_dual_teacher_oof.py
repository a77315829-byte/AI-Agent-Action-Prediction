from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from feature_utils_qwen_v4 import ACTION_TO_FAMILY, ALL_CLASSES

NUM_CLASSES = len(ALL_CLASSES)
FAMILY_INDEX = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        f1_score(
            labels,
            predictions,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0,
        )
    )


def final_logits(
    action_logits: np.ndarray,
    family_logits: np.ndarray,
    class_weights: np.ndarray,
    postprocess: dict,
) -> np.ndarray:
    return (
        action_logits.astype(np.float64)
        / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"])
        * family_logits.astype(np.float64)[:, FAMILY_INDEX]
        - float(postprocess["prior_beta"])
        * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )


def log_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a label-aware selective dual-teacher OOF package from "
            "the original current96 teacher and cur160 teacher."
        )
    )
    parser.add_argument("--original-oof", type=Path, required=True)
    parser.add_argument("--cur160-oof", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-weight", type=float, default=1.0)
    parser.add_argument("--both-correct-cur160-weight", type=float, default=0.5)
    args = parser.parse_args()

    if not 0.5 <= args.selected_weight <= 1.0:
        raise ValueError("--selected-weight must be in [0.5, 1.0].")
    if not 0.0 <= args.both_correct_cur160_weight <= 1.0:
        raise ValueError("--both-correct-cur160-weight must be in [0, 1].")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    original = np.load(args.original_oof)
    cur160 = np.load(args.cur160_oof)
    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))

    required = {"action_logits", "family_logits", "labels"}
    for name, package in (("original", original), ("cur160", cur160)):
        missing = required - set(package.files)
        if missing:
            raise RuntimeError(f"{name} OOF missing keys: {sorted(missing)}")

    labels_original = original["labels"].astype(np.int64)
    labels_cur160 = cur160["labels"].astype(np.int64)
    if not np.array_equal(labels_original, labels_cur160):
        raise RuntimeError("OOF labels are not aligned.")
    labels = labels_cur160

    original_action = original["action_logits"].astype(np.float64)
    original_family = original["family_logits"].astype(np.float64)
    cur160_action = cur160["action_logits"].astype(np.float64)
    cur160_family = cur160["family_logits"].astype(np.float64)

    if original_action.shape != cur160_action.shape:
        raise RuntimeError("Action-logit shapes differ.")
    if original_family.shape != cur160_family.shape:
        raise RuntimeError("Family-logit shapes differ.")

    class_weights = np.asarray(
        postprocess["training_class_weights"],
        dtype=np.float64,
    )

    original_final = final_logits(
        original_action, original_family, class_weights, postprocess
    )
    cur160_final = final_logits(
        cur160_action, cur160_family, class_weights, postprocess
    )

    original_pred = original_final.argmax(axis=1)
    cur160_pred = cur160_final.argmax(axis=1)

    original_correct = original_pred == labels
    cur160_correct = cur160_pred == labels

    original_only = original_correct & ~cur160_correct
    cur160_only = cur160_correct & ~original_correct
    both_correct = original_correct & cur160_correct
    both_wrong = ~original_correct & ~cur160_correct

    original_log_probs = log_softmax(original_final)
    cur160_log_probs = log_softmax(cur160_final)
    row_index = np.arange(len(labels))
    original_true_logp = original_log_probs[row_index, labels]
    cur160_true_logp = cur160_log_probs[row_index, labels]

    original_better_wrong = both_wrong & (
        original_true_logp >= cur160_true_logp
    )
    cur160_better_wrong = both_wrong & ~original_better_wrong

    cur160_weight = np.full(
        len(labels),
        args.both_correct_cur160_weight,
        dtype=np.float64,
    )
    cur160_weight[original_only] = 1.0 - args.selected_weight
    cur160_weight[cur160_only] = args.selected_weight
    cur160_weight[original_better_wrong] = 1.0 - args.selected_weight
    cur160_weight[cur160_better_wrong] = args.selected_weight

    original_weight = 1.0 - cur160_weight

    selected_action = (
        original_weight[:, None] * original_action
        + cur160_weight[:, None] * cur160_action
    )
    selected_family = (
        original_weight[:, None] * original_family
        + cur160_weight[:, None] * cur160_family
    )

    selected_final = final_logits(
        selected_action, selected_family, class_weights, postprocess
    )
    selected_pred = selected_final.argmax(axis=1)

    oracle_pred = original_pred.copy()
    oracle_pred[cur160_only] = cur160_pred[cur160_only]
    oracle_score = macro_f1(labels, oracle_pred)

    selected_score = macro_f1(labels, selected_pred)
    original_score = macro_f1(labels, original_pred)
    cur160_score = macro_f1(labels, cur160_pred)

    fold_ids = None
    if "fold_ids" in cur160.files:
        fold_ids = cur160["fold_ids"].astype(np.int8)
    elif "fold_ids" in original.files:
        fold_ids = original["fold_ids"].astype(np.int8)

    payload = {
        "action_logits": selected_action.astype(np.float32),
        "family_logits": selected_family.astype(np.float32),
        "labels": labels.astype(np.int64),
        "teacher_source_cur160_weight": cur160_weight.astype(np.float32),
    }
    if fold_ids is not None:
        payload["fold_ids"] = fold_ids

    output_npz = args.output_dir / "oof_logits_selective_dual_teacher.npz"
    np.savez_compressed(output_npz, **payload)

    report = {
        "original_macro_f1": original_score,
        "cur160_macro_f1": cur160_score,
        "selective_teacher_macro_f1": selected_score,
        "oracle_upper_bound_macro_f1": oracle_score,
        "selected_weight": float(args.selected_weight),
        "both_correct_cur160_weight": float(
            args.both_correct_cur160_weight
        ),
        "row_counts": {
            "original_only_correct": int(original_only.sum()),
            "cur160_only_correct": int(cur160_only.sum()),
            "both_correct": int(both_correct.sum()),
            "both_wrong": int(both_wrong.sum()),
            "both_wrong_original_higher_true_probability": int(
                original_better_wrong.sum()
            ),
            "both_wrong_cur160_higher_true_probability": int(
                cur160_better_wrong.sum()
            ),
        },
        "output_npz": str(output_npz),
    }

    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Original OOF Macro-F1:       {original_score:.6f}")
    print(f"Cur160 OOF Macro-F1:         {cur160_score:.6f}")
    print(f"Selective teacher Macro-F1:  {selected_score:.6f}")
    print(f"Oracle upper-bound Macro-F1: {oracle_score:.6f}")
    print()
    print("Routing rows:")
    for key, value in report["row_counts"].items():
        print(f"  {key}: {value}")
    print()
    print("Saved:", output_npz)
    print("Report:", args.output_dir / "report.json")


if __name__ == "__main__":
    main()
