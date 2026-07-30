"""
probe_v39_class_specific_blend.py

Class-specific constrained blend, screened with nested CV instead of a
full-validation-only weight sweep.

IMPORTANT CAVEAT: the candidate class groups (A/B/C below) were themselves
chosen by looking at which classes improved on THIS validation set in the
open-files tree run. That's a form of double-dipping we can't fully undo
without fresh data. What THIS script fixes is the weight selection step:
weight is picked on a TUNE split and scored on a disjoint EVAL split,
repeated over folds/seeds, so at least the weight itself isn't fit and
scored on the same rows. Treat any PROMISING verdict here as weaker
evidence than a normal nested-CV result, precisely because of the group
definition issue -- and remember V30: even strong nested-CV signals have
failed on public before.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils_qwen_v4 import ALL_CLASSES, ACTION_TO_FAMILY

NUM_CLASSES = len(ALL_CLASSES)
LABEL2ID = {label: i for i, label in enumerate(ALL_CLASSES)}

CANDIDATE_GROUPS = {
    "A": ["run_bash", "run_tests", "lint_or_typecheck"],
    "B": ["list_directory", "run_bash", "run_tests", "lint_or_typecheck"],
    "C": ["list_directory", "run_bash", "run_tests", "lint_or_typecheck", "plan_task", "web_search"],
}


def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def macro_f1(labels, predictions) -> float:
    return f1_score(labels, predictions, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0)


def blend_predict(qwen_proba, tree_proba, target_ids, weight):
    log_qwen = np.log(np.maximum(qwen_proba, 1e-12))
    log_tree = np.log(np.maximum(tree_proba, 1e-12))
    combined = log_qwen.copy()
    combined[:, target_ids] = combined[:, target_ids] + weight * log_tree[:, target_ids]
    return combined.argmax(axis=1)


def nested_cv_group(labels, qwen_proba, tree_proba, target_ids, weight_grid, n_folds, n_seeds):
    n = len(labels)
    fold_gains = []
    chosen_weights = []
    rng_master = np.random.default_rng(0)

    for seed in range(n_seeds):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        permutation = rng.permutation(n)
        folds = np.array_split(permutation, n_folds)

        for fold_index in range(n_folds):
            eval_idx = folds[fold_index]
            tune_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_index])

            best = None
            for w in weight_grid:
                pred = blend_predict(qwen_proba[tune_idx], tree_proba[tune_idx], target_ids, w)
                f1 = macro_f1(labels[tune_idx], pred)
                if best is None or f1 > best[0]:
                    best = (f1, w)
            best_weight = best[1]
            chosen_weights.append(best_weight)

            qwen_pred_eval = qwen_proba[eval_idx].argmax(axis=1)
            qwen_f1_eval = macro_f1(labels[eval_idx], qwen_pred_eval)
            blend_pred_eval = blend_predict(qwen_proba[eval_idx], tree_proba[eval_idx], target_ids, best_weight)
            blend_f1_eval = macro_f1(labels[eval_idx], blend_pred_eval)
            fold_gains.append(blend_f1_eval - qwen_f1_eval)

    fold_gains = np.asarray(fold_gains)
    return {
        "nested_cv_gain_mean": float(fold_gains.mean()),
        "nested_cv_gain_min": float(fold_gains.min()),
        "nested_cv_positive": int((fold_gains > 0).sum()),
        "nested_cv_total": len(fold_gains),
        "avg_chosen_weight": float(np.mean(chosen_weights)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-logits", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--tree-prob", type=Path, required=True, help="tree_prob_val.npy from the open-files probe")
    parser.add_argument("--weight-grid", type=float, nargs="+", default=[0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.qwen_logits)
    action_logits = npz["action_logits"].astype(np.float64)
    family_logits = npz["family_logits"].astype(np.float64)
    labels = npz["labels"].astype(np.int64)

    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))
    family_index = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)
    class_weights = np.asarray(postprocess["training_class_weights"], dtype=np.float64)
    final_logits = (
        action_logits / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"]) * family_logits[:, family_index]
        - float(postprocess["prior_beta"]) * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )
    qwen_proba = stable_softmax(final_logits)
    tree_proba = np.load(args.tree_prob)
    assert tree_proba.shape[0] == len(labels), f"tree_prob rows={tree_proba.shape[0]} vs labels={len(labels)}"

    qwen_pred = qwen_proba.argmax(axis=1)
    qwen_f1 = macro_f1(labels, qwen_pred)
    print(f"Qwen baseline Macro-F1: {qwen_f1:.6f}\n")

    all_results = {}
    for group_name, group_classes in CANDIDATE_GROUPS.items():
        target_ids = np.asarray([LABEL2ID[c] for c in group_classes])
        print(f"=== Group {group_name}: {group_classes} ===")

        # full-data reference (NOT the trust metric -- same data the groups were chosen from)
        best_full = None
        for w in args.weight_grid:
            pred = blend_predict(qwen_proba, tree_proba, target_ids, w)
            f1 = macro_f1(labels, pred)
            if best_full is None or f1 > best_full[0]:
                best_full = (f1, w)
        print(f"  Full-data best (reference only): weight={best_full[1]:.2f} f1={best_full[0]:.6f} gain={best_full[0]-qwen_f1:+.6f}")

        cv_results = nested_cv_group(labels, qwen_proba, tree_proba, target_ids, args.weight_grid, args.n_folds, args.n_seeds)
        print(f"  Nested CV ({args.n_folds}x{args.n_seeds}={args.n_folds*args.n_seeds}): "
              f"gain_mean={cv_results['nested_cv_gain_mean']:+.6f} "
              f"gain_min={cv_results['nested_cv_gain_min']:+.6f} "
              f"positive={cv_results['nested_cv_positive']}/{cv_results['nested_cv_total']} "
              f"avg_weight={cv_results['avg_chosen_weight']:.3f}")

        positive_rate = cv_results["nested_cv_positive"] / cv_results["nested_cv_total"]
        if positive_rate >= 0.8 and cv_results["nested_cv_gain_mean"] > 0.0005:
            verdict = "PROMISING (weak evidence -- group itself was validation-derived, see caveat)"
        elif positive_rate >= 0.5:
            verdict = "MARGINAL"
        else:
            verdict = "DISCARD"
        print(f"  Verdict: {verdict}\n")

        all_results[group_name] = {
            "classes": group_classes,
            "full_data_best": {"weight": best_full[1], "f1": best_full[0], "gain": best_full[0] - qwen_f1},
            **cv_results,
            "verdict": verdict,
        }

    if args.report:
        (args.output_dir / "v39_class_specific_report.json").write_text(
            json.dumps({"qwen_f1": qwen_f1, "groups": all_results}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved report to {args.output_dir}")


if __name__ == "__main__":
    main()
