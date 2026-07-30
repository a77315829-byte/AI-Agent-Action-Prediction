"""
probe_v25_class_specific_tree_weight.py

Generalizes V23's single scalar tree_blend_weight (0.25 for all classes) to
a per-class weight vector:

    blended_log_prob[:, c] = (1 - w[c]) * log(qwen_prob[:, c]) + w[c] * log(tree_prob[:, c])

Motivation: V23 heavy validation showed asymmetric per-class effects (some
classes gained from the tree blend, some lost). A flat weight is a
compromise across all of them; per-class weights should do better *if* the
per-class signal is real and not validation-set noise.

Why this is NOT another V19/V20/V22 (all of which looked great on full
validation and died on holdout): those tuned on 100% of the validation set
and only checked holdout afterward. Here the class-weight selection itself
is done with nested CV -- weights are picked only on "tune" folds and
scored only on folds never used for tuning, repeated over multiple fold
shuffles. If it still doesn't hold up, that's the answer, and V25 should be
discarded like the others.

Regularization (both are load-bearing, not cosmetic):
  --top-k-classes   only allow this many classes to deviate from the flat
                     baseline weight (ranked by |qwen_correct - tree_correct|
                     disagreement count on the tuning folds)
  --shrinkage        pull the greedily-chosen per-class weight back toward
                     the flat baseline by this fraction (0 = no shrink,
                     1 = fully back to baseline / no-op)

ASSUMPTIONS (same as probe_v24_calibrator_tree_blend.py -- adjust the
loading section if these don't match the real artifacts):
  - --qwen-logits npz has "action_logits", "family_logits" over the same
    rows as --val-indices.
  - --postprocess json has action_temperature, family_weight, prior_beta,
    training_class_weights.
  - --tree-prob is tree_prob_val.npy from probe_v23_tree_tfidf_structured.py,
    same row order as --val-indices.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils_qwen_v4 import ALL_CLASSES, ACTION_TO_FAMILY  # noqa: E402

NUM_CLASSES = len(ALL_CLASSES)
CANDIDATE_WEIGHTS = [0.10, 0.25, 0.40, 0.55]


def load_jsonl(path: Path) -> List[dict]:
    samples = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return f1_score(labels, predictions, average="macro")


def blend_predict(qwen_prob: np.ndarray, tree_prob: np.ndarray, weight_vector: np.ndarray) -> np.ndarray:
    log_p = (
        (1.0 - weight_vector)[None, :] * np.log(np.maximum(qwen_prob, 1e-12))
        + weight_vector[None, :] * np.log(np.maximum(tree_prob, 1e-12))
    )
    return log_p.argmax(axis=1)


def rank_classes_by_disagreement(
    qwen_prob: np.ndarray,
    tree_prob: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Rank classes by |qwen_correct_only - tree_correct_only| count, restricted
    to rows where the true label is that class. High-disagreement classes are
    where a class-specific weight has the most room to help or hurt."""
    qwen_pred = qwen_prob.argmax(axis=1)
    tree_pred = tree_prob.argmax(axis=1)

    scores = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() == 0:
            continue
        qwen_correct = (qwen_pred[mask] == c).sum()
        tree_correct = (tree_pred[mask] == c).sum()
        scores[c] = abs(int(qwen_correct) - int(tree_correct))
    return np.argsort(-scores)


def tune_class_weights(
    qwen_prob: np.ndarray,
    tree_prob: np.ndarray,
    labels: np.ndarray,
    baseline_weight: float,
    top_k: int,
    shrinkage: float,
) -> np.ndarray:
    """Greedy one-pass per-class weight search on the TUNE split only."""
    weight_vector = np.full(NUM_CLASSES, baseline_weight, dtype=np.float64)
    ranked_classes = rank_classes_by_disagreement(qwen_prob, tree_prob, labels)
    candidate_classes = ranked_classes[:top_k]

    current_pred = blend_predict(qwen_prob, tree_prob, weight_vector)
    current_f1 = macro_f1(labels, current_pred)

    for c in candidate_classes:
        best_weight = weight_vector[c]
        best_f1 = current_f1
        for candidate in CANDIDATE_WEIGHTS:
            trial_vector = weight_vector.copy()
            trial_vector[c] = candidate
            trial_pred = blend_predict(qwen_prob, tree_prob, trial_vector)
            trial_f1 = macro_f1(labels, trial_pred)
            if trial_f1 > best_f1:
                best_f1 = trial_f1
                best_weight = candidate
        # shrink the chosen deviation back toward baseline
        shrunk_weight = baseline_weight + (1.0 - shrinkage) * (best_weight - baseline_weight)
        weight_vector[c] = shrunk_weight
        current_pred = blend_predict(qwen_prob, tree_prob, weight_vector)
        current_f1 = macro_f1(labels, current_pred)

    return weight_vector


def nested_cv(
    qwen_prob: np.ndarray,
    tree_prob: np.ndarray,
    labels: np.ndarray,
    baseline_weight: float,
    top_k: int,
    shrinkage: float,
    n_folds: int,
    n_seeds: int,
) -> dict:
    n = len(labels)
    fold_gains = []
    tuned_vectors = []

    rng_master = np.random.default_rng(0)
    for seed in range(n_seeds):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        permutation = rng.permutation(n)
        folds = np.array_split(permutation, n_folds)

        for fold_index in range(n_folds):
            eval_idx = folds[fold_index]
            tune_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_index])

            weight_vector = tune_class_weights(
                qwen_prob[tune_idx], tree_prob[tune_idx], labels[tune_idx],
                baseline_weight, top_k, shrinkage,
            )
            tuned_vectors.append(weight_vector)

            baseline_vector = np.full(NUM_CLASSES, baseline_weight)
            baseline_pred_eval = blend_predict(qwen_prob[eval_idx], tree_prob[eval_idx], baseline_vector)
            v25_pred_eval = blend_predict(qwen_prob[eval_idx], tree_prob[eval_idx], weight_vector)

            baseline_f1_eval = macro_f1(labels[eval_idx], baseline_pred_eval)
            v25_f1_eval = macro_f1(labels[eval_idx], v25_pred_eval)
            fold_gains.append(v25_f1_eval - baseline_f1_eval)

    fold_gains = np.asarray(fold_gains)
    averaged_weight_vector = np.mean(tuned_vectors, axis=0)

    return {
        "nested_cv_gain_mean_vs_v23": float(fold_gains.mean()),
        "nested_cv_gain_min_vs_v23": float(fold_gains.min()),
        "nested_cv_positive": int((fold_gains > 0).sum()),
        "nested_cv_total": len(fold_gains),
        "averaged_weight_vector": averaged_weight_vector.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--qwen-logits", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--tree-prob", type=Path, required=True)
    parser.add_argument("--val-indices", type=Path, required=True)
    parser.add_argument("--baseline-weight", type=float, default=0.25)
    parser.add_argument("--top-k-classes", type=int, default=5)
    parser.add_argument("--shrinkage", type=float, default=0.5)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=3, help="total CV evaluations = n_folds * n_seeds")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = load_jsonl(args.data)
    val_indices = np.load(args.val_indices)
    samples = [all_samples[i] for i in val_indices]

    label_map = {}
    with args.labels_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            label_map[str(row["id"])] = row["action"]
    action_to_id = {name: i for i, name in enumerate(ALL_CLASSES)}
    labels = np.asarray(
        [action_to_id[label_map[str(s.get("id", ""))]] for s in samples],
        dtype=np.int64,
    )

    logits_npz = np.load(args.qwen_logits)
    action_logits = logits_npz["action_logits"].astype(np.float64)
    family_logits = logits_npz["family_logits"].astype(np.float64)
    assert action_logits.shape[0] == len(samples), (
        f"qwen-logits rows={action_logits.shape[0]} vs val samples={len(samples)} -- "
        "the npz is expected to already be validation-subset-only (do not index it with --val-indices)"
    )

    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))
    family_index = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)
    training_class_weights = np.asarray(postprocess["training_class_weights"], dtype=np.float64)

    final_logits = (
        action_logits / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"]) * family_logits[:, family_index]
        - float(postprocess["prior_beta"]) * np.log(np.maximum(training_class_weights, 1e-12))[None, :]
    )
    qwen_prob = stable_softmax(final_logits)
    tree_prob = np.load(args.tree_prob)
    assert tree_prob.shape[0] == len(samples), (
        f"tree_prob rows={tree_prob.shape[0]} vs val samples={len(samples)} -- "
        "confirm --tree-prob was produced on the same --val-indices"
    )

    # 1) Reference numbers: flat V23 weight on full validation.
    baseline_vector = np.full(NUM_CLASSES, args.baseline_weight)
    qwen_pred = qwen_prob.argmax(axis=1)
    v23_pred = blend_predict(qwen_prob, tree_prob, baseline_vector)
    qwen_f1 = macro_f1(labels, qwen_pred)
    v23_f1 = macro_f1(labels, v23_pred)

    # 2) Nested CV: pick per-class weights on tune folds, score on eval folds only.
    cv_results = nested_cv(
        qwen_prob, tree_prob, labels,
        args.baseline_weight, args.top_k_classes, args.shrinkage,
        args.n_folds, args.n_seeds,
    )

    # 3) For reference only (NOT the number to trust): full-validation gain of
    #    the CV-averaged weight vector, tuned on data it was also scored on.
    #    This is expected to look better than nested_cv numbers -- that gap
    #    IS the overfitting this design is trying to catch.
    averaged_vector = np.asarray(cv_results["averaged_weight_vector"])
    v25_full_pred = blend_predict(qwen_prob, tree_prob, averaged_vector)
    v25_full_f1 = macro_f1(labels, v25_full_pred)

    deviating_classes = [
        {"class": ALL_CLASSES[c], "weight": round(float(averaged_vector[c]), 3)}
        for c in range(NUM_CLASSES)
        if abs(averaged_vector[c] - args.baseline_weight) > 1e-6
    ]

    results = {
        "qwen_f1": qwen_f1,
        "v23_flat_weight_f1": v23_f1,
        "v25_full_validation_f1_DO_NOT_TRUST_ALONE": v25_full_f1,
        "v25_full_validation_gain_vs_v23_DO_NOT_TRUST_ALONE": v25_full_f1 - v23_f1,
        **cv_results,
        "deviating_classes": deviating_classes,
        "baseline_weight": args.baseline_weight,
        "top_k_classes": args.top_k_classes,
        "shrinkage": args.shrinkage,
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))

    positive_rate = cv_results["nested_cv_positive"] / cv_results["nested_cv_total"]
    if positive_rate >= 0.8:
        verdict = "PROMISING - nested CV positive rate >= 0.8, consider as submit candidate"
    elif positive_rate >= 0.5:
        verdict = "MARGINAL - mixed nested CV results, probably not worth the submission risk"
    else:
        verdict = "DISCARD - nested CV mostly negative, same failure mode as V19/V20/V22"
    print(f"\nVerdict: {verdict}")

    if args.report:
        results["verdict"] = verdict
        (args.output_dir / "v25_report.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved report to {args.output_dir / 'v25_report.json'}")


if __name__ == "__main__":
    main()
