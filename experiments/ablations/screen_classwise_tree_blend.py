from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils_qwen_v4 import ACTION_TO_FAMILY, ALL_CLASSES


N_CLASSES = len(ALL_CLASSES)
LABEL2ID = {name: i for i, name in enumerate(ALL_CLASSES)}
FAMILY_INDEX = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_labels(samples, path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        mapping = {
            str(row["id"]): LABEL2ID[str(row["action"])]
            for row in csv.DictReader(f)
        }
    return np.asarray(
        [mapping[str(sample["id"])] for sample in samples],
        dtype=np.int64,
    )


def macro_f1(y, pred):
    return float(
        f1_score(
            y,
            pred,
            labels=np.arange(N_CLASSES),
            average="macro",
            zero_division=0,
        )
    )


def stable_log_softmax(x):
    x = x.astype(np.float64, copy=False)
    shifted = x - x.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def load_qwen_logp(npz_path: Path, post_path: Path, expected_labels):
    pack = np.load(npz_path)
    labels = pack["labels"].astype(np.int64)
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError(f"Label alignment failed: {npz_path}")

    post = json.loads(post_path.read_text(encoding="utf-8"))
    class_weights = np.asarray(
        post["training_class_weights"],
        dtype=np.float64,
    )
    action = pack["action_logits"].astype(np.float64)
    family = pack["family_logits"].astype(np.float64)

    logits = (
        action / float(post["action_temperature"])
        + float(post["family_weight"]) * family[:, FAMILY_INDEX]
        - float(post["prior_beta"])
        * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )
    return stable_log_softmax(logits)


def predict(qwen_logp, tree_logp, weights, indices=None):
    if indices is not None:
        qwen_logp = qwen_logp[indices]
        tree_logp = tree_logp[indices]
    w = weights[None, :]
    score = (1.0 - w) * qwen_logp + w * tree_logp
    return score.argmax(axis=1).astype(np.int64)


def score_config(y, qwen_logp, tree_logp, indices, weights):
    return macro_f1(
        y[indices],
        predict(qwen_logp, tree_logp, weights, indices),
    )


def coordinate_descent(
    y,
    qwen_logp,
    tree_logp,
    tune_idx,
    grid,
    global_weight,
    passes,
    shrink_penalty,
):
    weights = np.full(N_CLASSES, global_weight, dtype=np.float64)

    def penalized(candidate):
        raw = score_config(
            y, qwen_logp, tree_logp, tune_idx, candidate
        )
        deviation = np.mean(
            np.abs(candidate - global_weight)
        ) / max(global_weight, 1e-12)
        return raw - shrink_penalty * deviation, raw

    current_raw = score_config(
        y, qwen_logp, tree_logp, tune_idx, weights
    )

    for pass_id in range(passes):
        changed = False
        for class_id in range(N_CLASSES):
            old = float(weights[class_id])
            best_weight = old
            best_pen = -1e100
            best_raw = current_raw

            for candidate_weight in grid:
                candidate = weights.copy()
                candidate[class_id] = float(candidate_weight)
                pen, raw = penalized(candidate)

                better = pen > best_pen + 1e-12
                tied = abs(pen - best_pen) <= 1e-12
                closer = (
                    abs(candidate_weight - global_weight)
                    < abs(best_weight - global_weight) - 1e-12
                )
                if better or (tied and closer):
                    best_pen = pen
                    best_raw = raw
                    best_weight = float(candidate_weight)

            weights[class_id] = best_weight
            current_raw = best_raw
            changed |= abs(best_weight - old) > 1e-12

        print(
            f"    pass={pass_id + 1}/{passes} "
            f"tune_f1={current_raw:.6f} "
            f"changed_classes="
            f"{int(np.sum(np.abs(weights-global_weight)>1e-12))}"
        )
        if not changed:
            break

    return weights, current_raw


def as_weight_dict(weights):
    return {
        name: float(weights[i])
        for i, name in enumerate(ALL_CLASSES)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--qwen-oof", type=Path, required=True)
    p.add_argument("--qwen-postprocess", type=Path, required=True)
    p.add_argument("--tree-oof", type=Path, required=True)
    p.add_argument("--student-fold0-logits", type=Path)
    p.add_argument("--student-postprocess", type=Path)
    p.add_argument("--global-weight", type=float, default=0.15)
    p.add_argument(
        "--weight-grid",
        type=float,
        nargs="+",
        default=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    )
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--shrink-penalty", type=float, default=0.00025)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Load data...")
    samples = load_jsonl(args.data)
    labels = load_labels(samples, args.labels_csv)
    groups = np.asarray(
        [str(s["id"]).rsplit("-step_", 1)[0] for s in samples],
        dtype=object,
    )

    qwen_logp = load_qwen_logp(
        args.qwen_oof,
        args.qwen_postprocess,
        labels,
    )

    tree = np.load(args.tree_oof).astype(np.float64)
    if tree.shape != (len(labels), N_CLASSES):
        raise RuntimeError(
            f"Bad tree shape: {tree.shape}; "
            f"expected {(len(labels), N_CLASSES)}"
        )
    tree /= np.maximum(tree.sum(axis=1, keepdims=True), 1e-12)
    tree_logp = np.log(np.maximum(tree, 1e-12))

    grid = np.asarray(
        sorted(set(float(x) for x in args.weight_grid)),
        dtype=np.float64,
    )
    if not np.any(np.isclose(grid, args.global_weight)):
        raise ValueError("weight-grid must include global-weight")

    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )
    splits = list(
        splitter.split(np.zeros(len(labels)), labels, groups)
    )

    global_weights = np.full(
        N_CLASSES, args.global_weight, dtype=np.float64
    )
    baseline_pred = predict(
        qwen_logp, tree_logp, global_weights
    )
    baseline_full = macro_f1(labels, baseline_pred)
    print(
        f"Baseline global tree weight={args.global_weight:.2f}: "
        f"{baseline_full:.6f}"
    )

    crossfit_pred = np.full(len(labels), -1, dtype=np.int64)
    outer_weights = []
    outer_records = []

    for fold, (tune_idx, eval_idx) in enumerate(splits):
        print()
        print(
            f"Outer fold {fold}: "
            f"tune={len(tune_idx)} eval={len(eval_idx)}"
        )
        weights, tune_score = coordinate_descent(
            labels,
            qwen_logp,
            tree_logp,
            tune_idx,
            grid,
            args.global_weight,
            args.passes,
            args.shrink_penalty,
        )
        outer_weights.append(weights)

        candidate = predict(
            qwen_logp, tree_logp, weights, eval_idx
        )
        crossfit_pred[eval_idx] = candidate
        base_local = baseline_pred[eval_idx]
        base_score = macro_f1(labels[eval_idx], base_local)
        candidate_score = macro_f1(labels[eval_idx], candidate)
        gain = candidate_score - base_score

        rec = {
            "fold": fold,
            "tune_macro_f1": tune_score,
            "baseline_eval_macro_f1": base_score,
            "candidate_eval_macro_f1": candidate_score,
            "gain": gain,
            "changed_rows": int(np.sum(candidate != base_local)),
            "weights": as_weight_dict(weights),
        }
        outer_records.append(rec)
        print(
            f"  eval_gain={gain:+.6f}, "
            f"changed={rec['changed_rows']}"
        )
        print("  non-default weights:")
        for name, value in rec["weights"].items():
            if abs(value - args.global_weight) > 1e-12:
                print(f"    {name:18s} {value:.2f}")

    if np.any(crossfit_pred < 0):
        raise RuntimeError("Unfilled cross-fitted predictions")

    gains = np.asarray(
        [r["gain"] for r in outer_records],
        dtype=np.float64,
    )
    crossfit_score = macro_f1(labels, crossfit_pred)

    consensus = np.median(
        np.stack(outer_weights, axis=0),
        axis=0,
    )
    consensus_score = macro_f1(
        labels,
        predict(qwen_logp, tree_logp, consensus),
    )

    print()
    print("Fit full-data reference weights...")
    full_idx = np.arange(len(labels), dtype=np.int64)
    full_weights, full_score = coordinate_descent(
        labels,
        qwen_logp,
        tree_logp,
        full_idx,
        grid,
        args.global_weight,
        args.passes,
        args.shrink_penalty,
    )

    student_result = None
    if args.student_fold0_logits is not None:
        if args.student_postprocess is None:
            raise ValueError(
                "--student-postprocess is required with "
                "--student-fold0-logits"
            )

        fold0_idx = splits[0][1]
        student_labels = labels[fold0_idx]
        student_qwen_logp = load_qwen_logp(
            args.student_fold0_logits,
            args.student_postprocess,
            student_labels,
        )
        student_tree_logp = tree_logp[fold0_idx]

        # Clean: fold-0 weights were learned only on folds 1-4.
        fold0_weights = outer_weights[0]
        student_base = predict(
            student_qwen_logp,
            student_tree_logp,
            global_weights,
        )
        student_candidate = predict(
            student_qwen_logp,
            student_tree_logp,
            fold0_weights,
        )
        student_base_score = macro_f1(
            student_labels, student_base
        )
        student_candidate_score = macro_f1(
            student_labels, student_candidate
        )
        student_result = {
            "weights_source": "outer_fold0_tune_only",
            "baseline_macro_f1": student_base_score,
            "candidate_macro_f1": student_candidate_score,
            "gain": student_candidate_score - student_base_score,
            "changed_rows": int(
                np.sum(student_candidate != student_base)
            ),
            "weights": as_weight_dict(fold0_weights),
        }

    nested = {
        "cross_fitted_macro_f1": crossfit_score,
        "cross_fitted_gain": crossfit_score - baseline_full,
        "mean_fold_gain": float(gains.mean()),
        "min_fold_gain": float(gains.min()),
        "max_fold_gain": float(gains.max()),
        "positive_folds": int(np.sum(gains > 0)),
        "total_folds": int(len(gains)),
    }

    promising = (
        nested["mean_fold_gain"] >= 0.001
        and nested["positive_folds"] >= 4
        and nested["min_fold_gain"] >= -0.0005
    )
    if student_result is not None:
        promising &= student_result["gain"] >= 0.001

    report = {
        "baseline": {
            "global_weight": args.global_weight,
            "macro_f1": baseline_full,
        },
        "nested": nested,
        "outer_records": outer_records,
        "student_fold0": student_result,
        "consensus_reference": {
            "macro_f1": consensus_score,
            "gain": consensus_score - baseline_full,
            "weights": as_weight_dict(consensus),
        },
        "full_data_reference": {
            "macro_f1": full_score,
            "gain": full_score - baseline_full,
            "weights": as_weight_dict(full_weights),
        },
        "verdict": "PROMISING" if promising else "DISCARD",
    }

    (args.output_dir / "classwise_tree_blend_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("FINAL RESULT")
    print("=" * 72)
    print("Nested:")
    print(json.dumps(nested, indent=2))
    print("Student fold0:")
    print(
        json.dumps(student_result, ensure_ascii=False, indent=2)
        if student_result is not None
        else "not evaluated"
    )
    print("Consensus reference:")
    print(
        json.dumps(
            report["consensus_reference"],
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Full-data reference:")
    print(
        json.dumps(
            report["full_data_reference"],
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Verdict:", report["verdict"])
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
