from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils_qwen_v4 import ACTION_TO_FAMILY, ALL_CLASSES


NUM_CLASSES = len(ALL_CLASSES)
START_INDEX = NUM_CLASSES
LABEL2ID = {name: index for index, name in enumerate(ALL_CLASSES)}
FAMILY_INDEX = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def session_id(sample_id: str) -> str:
    return str(sample_id).rsplit("-step_", 1)[0]


def load_labels(samples: Sequence[dict], path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as file:
        mapping = {
            str(row["id"]): LABEL2ID[str(row["action"])]
            for row in csv.DictReader(file)
        }
    return np.asarray(
        [mapping[str(sample["id"])] for sample in samples],
        dtype=np.int64,
    )


def get_last_action_id(sample: dict) -> int:
    history = sample.get("history")
    if not isinstance(history, list):
        return START_INDEX

    for item in reversed(history):
        if (
            isinstance(item, dict)
            and item.get("role") == "assistant_action"
        ):
            name = str(item.get("name", ""))
            return LABEL2ID.get(name, START_INDEX)

    return START_INDEX


def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


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


def load_probabilities(
    npz_path: Path,
    postprocess_path: Path,
    expected_labels: np.ndarray,
    tree_prob_path: Path | None,
    tree_weight: float,
) -> np.ndarray:
    package = np.load(npz_path)

    required = {"action_logits", "family_logits", "labels"}
    missing = required - set(package.files)
    if missing:
        raise RuntimeError(f"NPZ missing keys: {sorted(missing)}")

    package_labels = package["labels"].astype(np.int64)
    if not np.array_equal(package_labels, expected_labels):
        raise RuntimeError(
            "NPZ labels do not align with the selected data rows."
        )

    postprocess = json.loads(
        postprocess_path.read_text(encoding="utf-8")
    )
    class_weights = np.asarray(
        postprocess["training_class_weights"],
        dtype=np.float64,
    )

    action_logits = package["action_logits"].astype(np.float64)
    family_logits = package["family_logits"].astype(np.float64)

    final_logits = (
        action_logits / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"])
        * family_logits[:, FAMILY_INDEX]
        - float(postprocess["prior_beta"])
        * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )
    probabilities = stable_softmax(final_logits)

    if tree_prob_path is not None:
        tree_probabilities = np.load(tree_prob_path).astype(np.float64)
        if len(tree_probabilities) != len(probabilities):
            raise RuntimeError(
                f"Tree rows={len(tree_probabilities)} "
                f"but selected rows={len(probabilities)}"
            )
        blended_log = (
            (1.0 - tree_weight)
            * np.log(np.maximum(probabilities, 1e-12))
            + tree_weight
            * np.log(np.maximum(tree_probabilities, 1e-12))
        )
        probabilities = stable_softmax(blended_log)
        print(f"Applied tree blend weight={tree_weight:.3f}")

    return probabilities


def fit_last_action_prior(
    labels: np.ndarray,
    last_action_ids: np.ndarray,
    train_indices: np.ndarray,
    smoothing: float,
) -> Tuple[np.ndarray, np.ndarray]:
    conditional_counts = np.full(
        (NUM_CLASSES + 1, NUM_CLASSES),
        smoothing,
        dtype=np.float64,
    )
    global_counts = np.full(
        NUM_CLASSES,
        smoothing,
        dtype=np.float64,
    )

    for index in train_indices:
        previous = int(last_action_ids[index])
        current = int(labels[index])
        conditional_counts[previous, current] += 1.0
        global_counts[current] += 1.0

    conditional = (
        conditional_counts
        / conditional_counts.sum(axis=1, keepdims=True)
    )
    global_probability = global_counts / global_counts.sum()

    return (
        np.log(np.maximum(conditional, 1e-12)),
        np.log(np.maximum(global_probability, 1e-12)),
    )


def adjusted_predictions(
    probabilities: np.ndarray,
    last_action_ids: np.ndarray,
    indices: np.ndarray,
    log_conditional: np.ndarray,
    log_global: np.ndarray,
    lam: float,
    rho: float,
    margin_threshold: float,
) -> np.ndarray:
    local_probability = probabilities[indices]
    score = np.log(np.maximum(local_probability, 1e-12))

    prior_score = (
        log_conditional[last_action_ids[indices]]
        - rho * log_global[None, :]
    )
    adjusted_score = score + lam * prior_score

    baseline = score.argmax(axis=1)
    adjusted = adjusted_score.argmax(axis=1)

    if margin_threshold < 1.0:
        sorted_probability = np.sort(local_probability, axis=1)
        margin = (
            sorted_probability[:, -1]
            - sorted_probability[:, -2]
        )
        use_prior = margin <= margin_threshold
        result = baseline.copy()
        result[use_prior] = adjusted[use_prior]
        return result

    return adjusted


def reconstruct_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
):
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    return list(
        splitter.split(np.zeros(len(labels)), labels, groups)
    )


def print_last_action_distribution(last_action_ids: np.ndarray) -> None:
    names = ALL_CLASSES + ["<START_OR_UNKNOWN>"]
    counts = Counter(int(value) for value in last_action_ids)
    print("Last-action distribution:")
    for index, count in counts.most_common():
        print(f"  {names[index]:20s} {count:6d}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Leak-free row-local transition prior using the last assistant "
            "action already present in each sample's history. Unlike session "
            "Viterbi, this remains deployable when test has one row per session."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--qwen-oof", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--tree-prob", type=Path, default=None)
    parser.add_argument("--tree-weight", type=float, default=0.15)
    parser.add_argument("--validation-fold", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lambda-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
    )
    parser.add_argument(
        "--smoothing-grid",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument(
        "--rho-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="Subtract rho * global label log-prior from the transition prior.",
    )
    parser.add_argument(
        "--margin-grid",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30, 1.0],
        help="1.0 means apply to all rows; smaller values gate to low-margin rows.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Load data...")
    all_samples = load_jsonl(args.data)
    all_labels = load_labels(all_samples, args.labels_csv)
    groups = np.asarray(
        [session_id(str(sample["id"])) for sample in all_samples],
        dtype=object,
    )
    splits = reconstruct_splits(
        all_labels,
        groups,
        args.n_splits,
        args.seed,
    )
    all_last_action_ids = np.asarray(
        [get_last_action_id(sample) for sample in all_samples],
        dtype=np.int64,
    )
    print_last_action_distribution(all_last_action_ids)

    configs = list(
        product(
            args.smoothing_grid,
            args.lambda_grid,
            args.rho_grid,
            args.margin_grid,
        )
    )

    if args.validation_fold is not None:
        fold = int(args.validation_fold)
        if fold < 0 or fold >= args.n_splits:
            raise ValueError("--validation-fold is out of range.")

        train_indices, eval_indices = splits[fold]
        selected_labels = all_labels[eval_indices]

        probabilities = load_probabilities(
            args.qwen_oof,
            args.postprocess,
            selected_labels,
            args.tree_prob,
            args.tree_weight,
        )
        baseline_predictions = probabilities.argmax(axis=1)
        baseline_score = macro_f1(
            selected_labels,
            baseline_predictions,
        )
        print(
            f"Validation-fold mode: fold={fold}, "
            f"train={len(train_indices)}, eval={len(eval_indices)}"
        )
        print(f"Baseline Macro-F1: {baseline_score:.6f}")

        # Local arrays align to eval_indices; prior is fit on original full-data indices.
        local_last_actions = all_last_action_ids[eval_indices]
        local_indices = np.arange(len(eval_indices), dtype=np.int64)

        rows = []
        for smoothing, lam, rho, margin in configs:
            log_conditional, log_global = fit_last_action_prior(
                all_labels,
                all_last_action_ids,
                train_indices,
                smoothing,
            )
            predictions = adjusted_predictions(
                probabilities,
                local_last_actions,
                local_indices,
                log_conditional,
                log_global,
                lam,
                rho,
                margin,
            )
            score = macro_f1(selected_labels, predictions)
            changed = int(
                np.sum(predictions != baseline_predictions)
            )
            rows.append(
                {
                    "smoothing": smoothing,
                    "lambda": lam,
                    "rho": rho,
                    "margin": margin,
                    "macro_f1": score,
                    "gain": score - baseline_score,
                    "changed_rows": changed,
                }
            )

        rows.sort(key=lambda row: row["macro_f1"], reverse=True)
        best = rows[0]
        print()
        print("Best fold configuration:")
        print(json.dumps(best, indent=2))
        print("Top 15:")
        for row in rows[:15]:
            print(
                f"  s={row['smoothing']:.2f} "
                f"lam={row['lambda']:.2f} "
                f"rho={row['rho']:.2f} "
                f"margin={row['margin']:.2f} "
                f"gain={row['gain']:+.6f} "
                f"changed={row['changed_rows']}"
            )

        report = {
            "mode": "validation_fold",
            "fold": fold,
            "baseline_macro_f1": baseline_score,
            "best": best,
            "all_configs": rows,
        }
        (args.output_dir / "last_action_prior_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    if len(all_labels) != len(np.load(args.qwen_oof)["labels"]):
        raise RuntimeError(
            "Full OOF mode requires a 70,000-row OOF NPZ. "
            "Use --validation-fold for a fold-only NPZ."
        )

    probabilities = load_probabilities(
        args.qwen_oof,
        args.postprocess,
        all_labels,
        args.tree_prob,
        args.tree_weight,
    )
    baseline_predictions = probabilities.argmax(axis=1)
    baseline_full = macro_f1(all_labels, baseline_predictions)
    print(f"Baseline full OOF Macro-F1: {baseline_full:.6f}")
    print(f"Screen configs: {len(configs)}")

    predictions_by_config: Dict[Tuple[float, float, float, float], np.ndarray] = {
        config: np.full(len(all_labels), -1, dtype=np.int64)
        for config in configs
    }
    fold_ids = np.full(len(all_labels), -1, dtype=np.int8)

    for fold, (train_indices, eval_indices) in enumerate(splits):
        fold_ids[eval_indices] = fold
        print(
            f"Fold {fold + 1}/{args.n_splits}: "
            f"train={len(train_indices)} eval={len(eval_indices)}"
        )

        priors = {}
        for smoothing in args.smoothing_grid:
            priors[smoothing] = fit_last_action_prior(
                all_labels,
                all_last_action_ids,
                train_indices,
                smoothing,
            )

        for smoothing, lam, rho, margin in configs:
            log_conditional, log_global = priors[smoothing]
            predictions_by_config[
                (smoothing, lam, rho, margin)
            ][eval_indices] = adjusted_predictions(
                probabilities,
                all_last_action_ids,
                eval_indices,
                log_conditional,
                log_global,
                lam,
                rho,
                margin,
            )

    baseline_fold_scores = {}
    for fold in range(args.n_splits):
        indices = np.flatnonzero(fold_ids == fold)
        baseline_fold_scores[fold] = macro_f1(
            all_labels[indices],
            baseline_predictions[indices],
        )

    rows = []
    for config, predictions in predictions_by_config.items():
        smoothing, lam, rho, margin = config
        fold_gains = []
        for fold in range(args.n_splits):
            indices = np.flatnonzero(fold_ids == fold)
            score = macro_f1(
                all_labels[indices],
                predictions[indices],
            )
            fold_gains.append(
                score - baseline_fold_scores[fold]
            )

        full_score = macro_f1(all_labels, predictions)
        rows.append(
            {
                "smoothing": smoothing,
                "lambda": lam,
                "rho": rho,
                "margin": margin,
                "macro_f1": full_score,
                "gain": full_score - baseline_full,
                "fold_gains": fold_gains,
                "positive_folds": int(
                    sum(gain > 0 for gain in fold_gains)
                ),
                "worst_fold_gain": float(min(fold_gains)),
                "changed_rows": int(
                    np.sum(predictions != baseline_predictions)
                ),
            }
        )

    rows.sort(key=lambda row: row["macro_f1"], reverse=True)
    best = rows[0]

    # Leave-one-fold-out selection over cross-fitted predictions.
    nested_records = []
    chosen_configs = []
    for outer_fold in range(args.n_splits):
        tune_indices = np.flatnonzero(fold_ids != outer_fold)
        eval_indices = np.flatnonzero(fold_ids == outer_fold)

        selected_config = max(
            configs,
            key=lambda config: macro_f1(
                all_labels[tune_indices],
                predictions_by_config[config][tune_indices],
            ),
        )
        chosen_configs.append(selected_config)

        candidate_score = macro_f1(
            all_labels[eval_indices],
            predictions_by_config[selected_config][eval_indices],
        )
        baseline_score = macro_f1(
            all_labels[eval_indices],
            baseline_predictions[eval_indices],
        )
        nested_records.append(
            {
                "outer_fold": outer_fold,
                "config": {
                    "smoothing": selected_config[0],
                    "lambda": selected_config[1],
                    "rho": selected_config[2],
                    "margin": selected_config[3],
                },
                "gain": candidate_score - baseline_score,
            }
        )

    nested_gains = np.asarray(
        [record["gain"] for record in nested_records],
        dtype=np.float64,
    )
    nested_summary = {
        "mean_gain": float(nested_gains.mean()),
        "min_gain": float(nested_gains.min()),
        "positive_folds": int((nested_gains > 0).sum()),
        "total_folds": int(len(nested_gains)),
    }

    promising = (
        best["gain"] >= 0.001
        and best["positive_folds"] >= 4
        and best["worst_fold_gain"] >= -0.0005
        and nested_summary["mean_gain"] > 0
        and nested_summary["positive_folds"] >= 4
        and best["lambda"] > 0
    )

    print()
    print("=== Best full-OOF config ===")
    print(json.dumps(best, indent=2))
    print()
    print("=== Nested selection ===")
    print(json.dumps(nested_summary, indent=2))
    for record in nested_records:
        print(
            f"  fold={record['outer_fold']} "
            f"gain={record['gain']:+.6f} "
            f"config={record['config']}"
        )
    print()
    print("Verdict:", "PROMISING" if promising else "DISCARD")

    report = {
        "mode": "full_oof",
        "baseline_macro_f1": baseline_full,
        "best": best,
        "nested_summary": nested_summary,
        "nested_records": nested_records,
        "verdict": "PROMISING" if promising else "DISCARD",
        "top_configs": rows[:50],
    }
    (args.output_dir / "last_action_prior_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
