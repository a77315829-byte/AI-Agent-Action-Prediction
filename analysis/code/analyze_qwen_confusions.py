from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


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

GROUPS: Dict[str, List[int]] = {
    "file_search": [0, 1, 2, 3],
    "edit_write_patch": [4, 5, 6],
    "exec_check": [7, 8, 9],
    "ask_plan": [10, 11],
}

DEFAULT_ACTION_TO_FAMILY = np.asarray(
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 3],
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze V12 Qwen validation confusion, top-k oracle, and group-level specialist potential."
    )
    parser.add_argument(
        "--logits",
        type=Path,
        default=Path("model/qwen_distill_v12_eval/validation_logits_v12.npz"),
    )
    parser.add_argument(
        "--postprocess",
        type=Path,
        default=Path("model/qwen_distill_v12_eval/postprocess.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/qwen_v12_confusion_analysis"),
    )
    parser.add_argument("--action-temperature", type=float, default=0.6)
    parser.add_argument("--family-weight", type=float, default=0.15)
    parser.add_argument("--prior-beta", type=float, default=0.25)
    parser.add_argument("--grid-search", action="store_true")
    return parser.parse_args()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=np.arange(len(CLASSES)),
            average="macro",
            zero_division=0,
        )
    )


def final_logits(
    action_logits: np.ndarray,
    family_logits: np.ndarray,
    action_to_family: np.ndarray,
    class_weights: np.ndarray,
    action_temperature: float,
    family_weight: float,
    prior_beta: float,
) -> np.ndarray:
    if action_temperature <= 0:
        raise ValueError("action_temperature must be positive.")

    return (
        action_logits.astype(np.float64) / action_temperature
        + family_weight * family_logits.astype(np.float64)[:, action_to_family]
        - prior_beta
        * np.log(np.maximum(class_weights.astype(np.float64), 1e-12))[None, :]
    )


def grid_search(
    action_logits: np.ndarray,
    family_logits: np.ndarray,
    labels: np.ndarray,
    action_to_family: np.ndarray,
    class_weights: np.ndarray,
) -> dict:
    best = {
        "macro_f1": -1.0,
        "action_temperature": None,
        "family_weight": None,
        "prior_beta": None,
    }

    temperatures = np.round(np.arange(0.40, 0.801, 0.05), 2)
    family_weights = np.round(np.arange(0.00, 0.401, 0.05), 2)
    prior_betas = np.round(np.arange(-0.25, 0.501, 0.05), 2)

    for temperature in temperatures:
        scaled_action = action_logits.astype(np.float64) / float(temperature)

        for family_weight in family_weights:
            family_adjustment = (
                float(family_weight)
                * family_logits.astype(np.float64)[:, action_to_family]
            )

            for prior_beta in prior_betas:
                logits = (
                    scaled_action
                    + family_adjustment
                    - float(prior_beta)
                    * np.log(np.maximum(class_weights, 1e-12))[None, :]
                )
                predictions = logits.argmax(axis=1)
                score = macro_f1(labels, predictions)

                if score > best["macro_f1"]:
                    best = {
                        "macro_f1": float(score),
                        "action_temperature": float(temperature),
                        "family_weight": float(family_weight),
                        "prior_beta": float(prior_beta),
                    }

    return best


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.logits)
    required = {"action_logits", "family_logits", "labels"}
    missing = required - set(npz.files)
    if missing:
        raise KeyError(f"Missing NPZ keys: {sorted(missing)}")

    action_logits = npz["action_logits"].astype(np.float32)
    family_logits = npz["family_logits"].astype(np.float32)
    labels = npz["labels"].astype(np.int64)

    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))
    action_to_family = np.asarray(
        postprocess.get("action_to_family", DEFAULT_ACTION_TO_FAMILY),
        dtype=np.int64,
    )
    class_weights = np.asarray(
        postprocess["training_class_weights"],
        dtype=np.float64,
    )

    if action_logits.shape != (len(labels), len(CLASSES)):
        raise ValueError(f"Unexpected action_logits shape: {action_logits.shape}")
    if family_logits.shape[0] != len(labels):
        raise ValueError("family_logits and labels length differ.")
    if len(class_weights) != len(CLASSES):
        raise ValueError(
            f"Expected {len(CLASSES)} training class weights, got {len(class_weights)}."
        )

    logits = final_logits(
        action_logits,
        family_logits,
        action_to_family,
        class_weights,
        args.action_temperature,
        args.family_weight,
        args.prior_beta,
    )
    predictions = logits.argmax(axis=1).astype(np.int64)
    baseline_score = macro_f1(labels, predictions)

    sorted_indices = np.argsort(logits, axis=1)[:, ::-1]
    top1 = sorted_indices[:, 0]
    top2 = sorted_indices[:, :2]
    top3 = sorted_indices[:, :3]

    top2_contains_truth = (top2 == labels[:, None]).any(axis=1)
    top3_contains_truth = (top3 == labels[:, None]).any(axis=1)
    wrong = predictions != labels

    top1_logits = np.take_along_axis(logits, top1[:, None], axis=1)[:, 0]
    second_logits = np.take_along_axis(logits, top2[:, 1:2], axis=1)[:, 0]
    margin = top1_logits - second_logits

    top2_oracle_predictions = predictions.copy()
    top2_oracle_predictions[top2_contains_truth] = labels[top2_contains_truth]

    top3_oracle_predictions = predictions.copy()
    top3_oracle_predictions[top3_contains_truth] = labels[top3_contains_truth]

    results = {
        "config": {
            "action_temperature": args.action_temperature,
            "family_weight": args.family_weight,
            "prior_beta": args.prior_beta,
        },
        "baseline_macro_f1": baseline_score,
        "samples": int(len(labels)),
        "errors": int(wrong.sum()),
        "accuracy": float((predictions == labels).mean()),
        "top2_truth_rate_all": float(top2_contains_truth.mean()),
        "top2_truth_rate_errors": float(top2_contains_truth[wrong].mean()),
        "top2_truth_count_errors": int(top2_contains_truth[wrong].sum()),
        "top3_truth_rate_errors": float(top3_contains_truth[wrong].mean()),
        "top2_oracle_macro_f1": macro_f1(labels, top2_oracle_predictions),
        "top2_oracle_gain": macro_f1(labels, top2_oracle_predictions) - baseline_score,
        "top3_oracle_macro_f1": macro_f1(labels, top3_oracle_predictions),
        "top3_oracle_gain": macro_f1(labels, top3_oracle_predictions) - baseline_score,
        "groups": {},
        "margin_buckets": [],
    }

    for group_name, ids_list in GROUPS.items():
        ids = np.asarray(ids_list, dtype=np.int64)

        internal_error_mask = (
            np.isin(labels, ids)
            & np.isin(predictions, ids)
            & wrong
        )
        truth_group_error_mask = np.isin(labels, ids) & wrong

        internal_oracle = predictions.copy()
        internal_oracle[internal_error_mask] = labels[internal_error_mask]

        truth_group_oracle = predictions.copy()
        truth_group_oracle[truth_group_error_mask] = labels[truth_group_error_mask]

        group_result = {
            "class_ids": ids_list,
            "classes": [CLASSES[index] for index in ids_list],
            "internal_errors": int(internal_error_mask.sum()),
            "internal_error_top2_rate": (
                float(top2_contains_truth[internal_error_mask].mean())
                if internal_error_mask.any()
                else 0.0
            ),
            "internal_oracle_macro_f1": macro_f1(labels, internal_oracle),
            "internal_oracle_gain": macro_f1(labels, internal_oracle) - baseline_score,
            "all_truth_errors": int(truth_group_error_mask.sum()),
            "all_truth_oracle_macro_f1": macro_f1(labels, truth_group_oracle),
            "all_truth_oracle_gain": macro_f1(labels, truth_group_oracle) - baseline_score,
        }
        results["groups"][group_name] = group_result

    margin_edges = [
        (0.00, 0.05),
        (0.05, 0.10),
        (0.10, 0.20),
        (0.20, 0.40),
        (0.40, 0.80),
        (0.80, 1.50),
        (1.50, float("inf")),
    ]

    for lower, upper in margin_edges:
        mask = (margin >= lower) & (margin < upper)
        error_count = int((mask & wrong).sum())
        bucket_count = int(mask.sum())

        oracle_mask = mask & wrong & top2_contains_truth
        oracle_predictions = predictions.copy()
        oracle_predictions[oracle_mask] = labels[oracle_mask]

        results["margin_buckets"].append(
            {
                "lower": lower,
                "upper": None if np.isinf(upper) else upper,
                "samples": bucket_count,
                "errors": error_count,
                "accuracy": (
                    float((predictions[mask] == labels[mask]).mean())
                    if bucket_count
                    else None
                ),
                "top2_recoverable_errors": int(oracle_mask.sum()),
                "top2_oracle_gain_if_bucket_only": (
                    macro_f1(labels, oracle_predictions) - baseline_score
                ),
            }
        )

    if args.grid_search:
        results["grid_search"] = grid_search(
            action_logits,
            family_logits,
            labels,
            action_to_family,
            class_weights,
        )

    confusion = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(len(CLASSES)),
    )
    np.savetxt(
        args.output_dir / "confusion_matrix.csv",
        confusion,
        fmt="%d",
        delimiter=",",
        header=",".join(CLASSES),
        comments="",
    )

    report = classification_report(
        labels,
        predictions,
        labels=np.arange(len(CLASSES)),
        target_names=CLASSES,
        digits=6,
        zero_division=0,
    )
    (args.output_dir / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    error_rows = []
    for index in np.flatnonzero(wrong):
        error_rows.append(
            {
                "validation_row": int(index),
                "true_id": int(labels[index]),
                "true_action": CLASSES[int(labels[index])],
                "pred_id": int(predictions[index]),
                "pred_action": CLASSES[int(predictions[index])],
                "second_id": int(top2[index, 1]),
                "second_action": CLASSES[int(top2[index, 1])],
                "truth_in_top2": bool(top2_contains_truth[index]),
                "margin": float(margin[index]),
            }
        )

    with (args.output_dir / "errors.jsonl").open("w", encoding="utf-8") as file:
        for row in error_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.output_dir / "analysis.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("V12 CONFUSION / ORACLE ANALYSIS")
    print("=" * 80)
    print(f"Macro-F1:              {baseline_score:.9f}")
    print(f"Errors:                {wrong.sum()} / {len(labels)}")
    print(
        f"Truth in top-2 errors: {top2_contains_truth[wrong].sum()} / {wrong.sum()} "
        f"({top2_contains_truth[wrong].mean():.2%})"
    )
    print(
        f"Top-2 oracle:          {results['top2_oracle_macro_f1']:.9f} "
        f"({results['top2_oracle_gain']:+.9f})"
    )
    print()

    for group_name, group_result in results["groups"].items():
        print(
            f"{group_name:20s} "
            f"internal_errors={group_result['internal_errors']:4d} "
            f"top2={group_result['internal_error_top2_rate']:.2%} "
            f"internal_oracle_gain={group_result['internal_oracle_gain']:+.6f} "
            f"all_truth_oracle_gain={group_result['all_truth_oracle_gain']:+.6f}"
        )

    if "grid_search" in results:
        best = results["grid_search"]
        print()
        print(
            "Grid best: "
            f"{best['macro_f1']:.9f} "
            f"T={best['action_temperature']} "
            f"family={best['family_weight']} "
            f"prior={best['prior_beta']}"
        )

    print()
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
