"""
probe_v29_three_way_blend.py

qwen + tree + structured-only three-way log-prob blend, screened with
nested-CV grid search (weights are picked on a TUNE split and scored on a
disjoint EVAL split, repeated over multiple fold/seed draws) -- same
discipline as probe_v25_class_specific_tree_weight.py, because a 2D grid
search on full validation is exactly the kind of setup that produced
false positives before (V19/V20/V22).

All three probability arrays must be OOF (leak-free) and in the SAME row
order as data/train.jsonl (i.e. what --qwen-oof's "labels" key uses).
  --tree-oof-proba        model/v27_tree_teacher_oof_light/tree_prob_oof_all.npy
                           (or the heavy version once you have it)
  --structured-oof-proba  model/v29_structured_screen/candidate_oof_proba.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

NUM_CLASSES_DEFAULT = 14


def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def macro_f1(labels, predictions, num_classes) -> float:
    return f1_score(labels, predictions, labels=np.arange(num_classes), average="macro", zero_division=0)


def blend_predict(qwen_p, tree_p, struct_p, w_tree, w_struct):
    w_qwen = 1.0 - w_tree - w_struct
    log_p = (
        w_qwen * np.log(np.maximum(qwen_p, 1e-12))
        + w_tree * np.log(np.maximum(tree_p, 1e-12))
        + w_struct * np.log(np.maximum(struct_p, 1e-12))
    )
    return log_p.argmax(axis=1)


def grid_search(qwen_p, tree_p, struct_p, labels, num_classes, tree_grid, struct_grid):
    best = None
    for w_tree in tree_grid:
        for w_struct in struct_grid:
            if w_tree + w_struct >= 1.0:
                continue
            pred = blend_predict(qwen_p, tree_p, struct_p, w_tree, w_struct)
            f1 = macro_f1(labels, pred, num_classes)
            if best is None or f1 > best[0]:
                best = (f1, w_tree, w_struct)
    return best


def nested_cv(
    qwen_p, tree_p, struct_p, labels, num_classes,
    tree_grid, struct_grid, n_folds, n_seeds,
):
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

            _, w_tree, w_struct = grid_search(
                qwen_p[tune_idx], tree_p[tune_idx], struct_p[tune_idx], labels[tune_idx],
                num_classes, tree_grid, struct_grid,
            )
            chosen_weights.append((w_tree, w_struct))

            qwen_pred_eval = qwen_p[eval_idx].argmax(axis=1)
            qwen_f1_eval = macro_f1(labels[eval_idx], qwen_pred_eval, num_classes)

            blend_pred_eval = blend_predict(
                qwen_p[eval_idx], tree_p[eval_idx], struct_p[eval_idx], w_tree, w_struct
            )
            blend_f1_eval = macro_f1(labels[eval_idx], blend_pred_eval, num_classes)
            fold_gains.append(blend_f1_eval - qwen_f1_eval)

    fold_gains = np.asarray(fold_gains)
    avg_w_tree = float(np.mean([w[0] for w in chosen_weights]))
    avg_w_struct = float(np.mean([w[1] for w in chosen_weights]))

    return {
        "nested_cv_gain_mean_vs_qwen": float(fold_gains.mean()),
        "nested_cv_gain_min_vs_qwen": float(fold_gains.min()),
        "nested_cv_positive": int((fold_gains > 0).sum()),
        "nested_cv_total": len(fold_gains),
        "avg_chosen_tree_weight": avg_w_tree,
        "avg_chosen_struct_weight": avg_w_struct,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-oof", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--tree-oof-proba", type=Path, required=True)
    parser.add_argument("--structured-oof-proba", type=Path, required=True)
    parser.add_argument("--tree-weight-grid", type=float, nargs="+", default=[0.0, 0.10, 0.20, 0.25, 0.35])
    parser.add_argument("--struct-weight-grid", type=float, nargs="+", default=[0.0, 0.10, 0.15, 0.20, 0.25])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    oof = np.load(args.qwen_oof)
    qwen_action = oof["action_logits"].astype(np.float64)
    qwen_family = oof["family_logits"].astype(np.float64)
    labels = oof["labels"].astype(np.int64)
    num_classes = int(labels.max()) + 1

    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from feature_utils_qwen_v4 import ACTION_TO_FAMILY
    family_index = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)
    class_weights = np.asarray(postprocess["training_class_weights"], dtype=np.float64)
    qwen_final = (
        qwen_action / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"]) * qwen_family[:, family_index]
        - float(postprocess["prior_beta"]) * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )
    qwen_p = stable_softmax(qwen_final)

    tree_p = np.load(args.tree_oof_proba)
    struct_p = np.load(args.structured_oof_proba)

    assert tree_p.shape[0] == len(labels), f"tree_oof rows={tree_p.shape[0]} vs labels={len(labels)}"
    assert struct_p.shape[0] == len(labels), f"structured_oof rows={struct_p.shape[0]} vs labels={len(labels)}"

    qwen_pred = qwen_p.argmax(axis=1)
    qwen_f1 = macro_f1(labels, qwen_pred, num_classes)

    # Reference numbers on full data -- NOT the trust metric, same caveat as always.
    tree_only_pred = blend_predict(qwen_p, tree_p, struct_p, 0.25, 0.0)
    tree_only_f1 = macro_f1(labels, tree_only_pred, num_classes)
    struct_only_pred = blend_predict(qwen_p, tree_p, struct_p, 0.0, 0.20)
    struct_only_f1 = macro_f1(labels, struct_only_pred, num_classes)

    full_best_f1, full_best_tree_w, full_best_struct_w = grid_search(
        qwen_p, tree_p, struct_p, labels, num_classes, args.tree_weight_grid, args.struct_weight_grid,
    )

    print("=== Full-data reference (NOT the trust metric) ===")
    print(f"qwen_f1={qwen_f1:.6f}")
    print(f"qwen+tree only (w=0.25): {tree_only_f1:.6f} (gain {tree_only_f1 - qwen_f1:+.6f})")
    print(f"qwen+structured only (w=0.20): {struct_only_f1:.6f} (gain {struct_only_f1 - qwen_f1:+.6f})")
    print(f"best 3-way on full data: tree_w={full_best_tree_w:.2f} struct_w={full_best_struct_w:.2f} f1={full_best_f1:.6f} (gain {full_best_f1 - qwen_f1:+.6f})")

    print()
    print(f"=== Nested CV ({args.n_folds} folds x {args.n_seeds} seeds = {args.n_folds * args.n_seeds} evals) ===")
    cv_results = nested_cv(
        qwen_p, tree_p, struct_p, labels, num_classes,
        args.tree_weight_grid, args.struct_weight_grid, args.n_folds, args.n_seeds,
    )
    print(json.dumps(cv_results, indent=2))

    positive_rate = cv_results["nested_cv_positive"] / cv_results["nested_cv_total"]
    if positive_rate >= 0.8:
        verdict = "PROMISING - 3-way blend beats qwen+tree alone on nested CV, worth building the full submission pipeline"
    elif positive_rate >= 0.5:
        verdict = "MARGINAL - inconsistent, probably not worth the extra moving part over qwen+tree alone"
    else:
        verdict = "DISCARD - structured candidate doesn't add value once tree is already in the blend"
    print(f"\nVerdict: {verdict}")

    if args.report:
        results = {
            "qwen_f1": qwen_f1,
            "tree_only_f1": tree_only_f1,
            "struct_only_f1": struct_only_f1,
            "full_best_tree_weight": full_best_tree_w,
            "full_best_struct_weight": full_best_struct_w,
            "full_best_f1": full_best_f1,
            **cv_results,
            "verdict": verdict,
        }
        (args.output_dir / "v29_three_way_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved report to {args.output_dir / 'v29_three_way_report.json'}")


if __name__ == "__main__":
    main()
