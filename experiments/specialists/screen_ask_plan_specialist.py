"""
screen_ask_plan_specialist.py

Corrected, single-script pipeline for the margin-gated file/search specialist.

Design fix vs a naive version: the specialist is TRAINED only on rows whose
true label is in {read_file, grep_search, list_directory, glob_pattern} (that's
the only sensible training target for a 4-way classifier), but its OOF
predictions are generated for EVERY row in each fold's held-out split --
because at real inference time you don't know the true label, so the
specialist must be able to score any row that gets gated to it (including
rows where Qwen's top1 guess landed in the group but the true label
actually wasn't -- those are exactly the risky cases a blend needs to survive).

Pipeline per fold:
  1. Split ALL rows (StratifiedGroupKFold on true label, grouped by session).
  2. Fit the 4-way specialist on TRAIN rows whose true label is in the group.
  3. Score EVERY row in the fold's VALIDATION split (not just group-true ones).
  4. Collect into a full (N, 4) OOF array.

Then: margin-gated log-prob blend into Qwen's full 14-class prediction,
screened with the same nested-CV discipline used throughout this project
(weights/threshold picked on a TUNE split, scored on a disjoint EVAL split).

Reminder (V30 lesson): even a nested-CV-validated gain here is not a public
guarantee. Treat any PROMISING verdict as "worth one careful public
submission", not "definitely works".
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils_qwen_v4 import ALL_CLASSES, ACTION_TO_FAMILY, build_structured_features, build_segments

NUM_CLASSES = len(ALL_CLASSES)
LABEL2ID = {label: i for i, label in enumerate(ALL_CLASSES)}
GROUP = ["ask_user", "plan_task"]
GROUP_IDS = np.asarray([LABEL2ID[c] for c in GROUP])
GROUP_LOCAL_ID = {global_id: local_id for local_id, global_id in enumerate(GROUP_IDS.tolist())}
NUM_GROUP_CLASSES = len(GROUP_IDS)


def load_jsonl(path: Path) -> List[dict]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_text(sample: dict) -> str:
    segments = build_segments(sample)
    return " ".join([segments["current"], segments["action"], segments["history"]])


def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def macro_f1(labels, predictions) -> float:
    return f1_score(labels, predictions, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0)


def build_specialist_oof(
    samples: List[dict],
    all_labels: np.ndarray,
    word_features: int,
    char_features: int,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    n = len(samples)
    groups = np.asarray([str(s["id"]).rsplit("-step_", 1)[0] for s in samples])
    print("Build specialist text/structured features for all rows...")
    texts = [build_text(s) for s in samples]
    structured = np.stack([build_structured_features(s) for s in samples]).astype(np.float32)

    oof_proba = np.zeros((n, NUM_GROUP_CLASSES), dtype=np.float64)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(np.zeros(n), all_labels, groups), start=1
    ):
        train_group_mask = np.isin(all_labels[train_idx], GROUP_IDS)
        train_fit_idx = train_idx[train_group_mask]
        train_labels_local = np.asarray([GROUP_LOCAL_ID[all_labels[i]] for i in train_fit_idx], dtype=np.int64)

        word_vec = TfidfVectorizer(max_features=word_features, ngram_range=(1, 2), min_df=2, max_df=0.98)
        char_vec = TfidfVectorizer(max_features=char_features, analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.98)

        fit_texts = [texts[i] for i in train_fit_idx]
        word_train = word_vec.fit_transform(fit_texts)
        char_train = char_vec.fit_transform(fit_texts)
        x_train = sparse.hstack([word_train, char_train, sparse.csr_matrix(structured[train_fit_idx])]).tocsr()

        model = LogisticRegression(solver="saga", max_iter=2000, C=1.0)
        model.fit(x_train, train_labels_local)

        # score EVERY validation row, not just group-true ones
        val_texts = [texts[i] for i in val_idx]
        word_val = word_vec.transform(val_texts)
        char_val = char_vec.transform(val_texts)
        x_val = sparse.hstack([word_val, char_val, sparse.csr_matrix(structured[val_idx])]).tocsr()
        local_proba = model.predict_proba(x_val)

        full_proba = np.zeros((len(val_idx), NUM_GROUP_CLASSES))
        full_proba[:, model.classes_.astype(int)] = local_proba
        oof_proba[val_idx] = full_proba

        group_true_val_mask = np.isin(all_labels[val_idx], GROUP_IDS)
        if group_true_val_mask.any():
            val_labels_local = np.asarray(
                [GROUP_LOCAL_ID[all_labels[i]] for i in val_idx[group_true_val_mask]], dtype=np.int64
            )
            fold_f1 = macro_f1_4way = f1_score(
                val_labels_local,
                full_proba[group_true_val_mask].argmax(axis=1),
                labels=np.arange(NUM_GROUP_CLASSES), average="macro", zero_division=0,
            )
            print(f"  Fold {fold}/{n_splits}: {NUM_GROUP_CLASSES}-way macro_f1 (on true-group val rows)={fold_f1:.6f}")

    return oof_proba


def blend_predict(qwen_proba, specialist_full_proba, eligible_mask, weight):
    """specialist_full_proba: (N, 4) aligned to GROUP_IDS order."""
    log_qwen = np.log(np.maximum(qwen_proba, 1e-12))
    prediction = log_qwen.argmax(axis=1)

    if not eligible_mask.any():
        return prediction

    log_spec = np.log(np.maximum(specialist_full_proba[eligible_mask], 1e-12))
    blended_group_log = (1 - weight) * log_qwen[np.ix_(eligible_mask, GROUP_IDS)] + weight * log_spec

    combined_log = log_qwen[eligible_mask].copy()
    combined_log[:, GROUP_IDS] = blended_group_log
    prediction[eligible_mask] = combined_log.argmax(axis=1)
    return prediction


def nested_cv_screen(
    labels, qwen_proba, specialist_full_proba, margin,
    margin_grid, weight_grid, n_folds, n_seeds,
):
    n = len(labels)
    fold_gains = []
    chosen = []
    rng_master = np.random.default_rng(0)

    top1_in_group = np.isin(qwen_proba.argmax(axis=1), GROUP_IDS)

    for seed in range(n_seeds):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        permutation = rng.permutation(n)
        folds = np.array_split(permutation, n_folds)

        for fold_index in range(n_folds):
            eval_idx = folds[fold_index]
            tune_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_index])

            best = None
            for m in margin_grid:
                eligible_tune = top1_in_group[tune_idx] & (margin[tune_idx] < m)
                for w in weight_grid:
                    pred = blend_predict(qwen_proba[tune_idx], specialist_full_proba[tune_idx], eligible_tune, w)
                    f1 = macro_f1(labels[tune_idx], pred)
                    if best is None or f1 > best[0]:
                        best = (f1, m, w)
            _, best_margin, best_weight = best
            chosen.append((best_margin, best_weight))

            eligible_eval = top1_in_group[eval_idx] & (margin[eval_idx] < best_margin)
            qwen_pred_eval = qwen_proba[eval_idx].argmax(axis=1)
            qwen_f1_eval = macro_f1(labels[eval_idx], qwen_pred_eval)
            blend_pred_eval = blend_predict(qwen_proba[eval_idx], specialist_full_proba[eval_idx], eligible_eval, best_weight)
            blend_f1_eval = macro_f1(labels[eval_idx], blend_pred_eval)
            fold_gains.append(blend_f1_eval - qwen_f1_eval)

    fold_gains = np.asarray(fold_gains)
    return {
        "nested_cv_gain_mean": float(fold_gains.mean()),
        "nested_cv_gain_min": float(fold_gains.min()),
        "nested_cv_positive": int((fold_gains > 0).sum()),
        "nested_cv_total": len(fold_gains),
        "avg_chosen_margin": float(np.mean([c[0] for c in chosen])),
        "avg_chosen_weight": float(np.mean([c[1] for c in chosen])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--qwen-oof", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--word-features", type=int, default=20000)
    parser.add_argument("--char-features", type=int, default=20000)
    parser.add_argument("--n-splits", type=int, default=5, help="folds for specialist OOF generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin-grid", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.3])
    parser.add_argument("--weight-grid", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--n-folds", type=int, default=5, help="nested-CV eval folds")
    parser.add_argument("--n-seeds", type=int, default=3, help="nested-CV eval seeds")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Load data...")
    samples = load_jsonl(args.data)
    with args.labels_csv.open(encoding="utf-8", newline="") as f:
        label_map = {str(row["id"]): str(row["action"]) for row in csv.DictReader(f)}
    all_labels = np.asarray([LABEL2ID[label_map[str(s["id"])]] for s in samples], dtype=np.int64)

    print("Load Qwen OOF...")
    oof = np.load(args.qwen_oof)
    action_logits = oof["action_logits"].astype(np.float64)
    family_logits = oof["family_logits"].astype(np.float64)
    oof_labels = oof["labels"].astype(np.int64)
    if not np.array_equal(all_labels, oof_labels):
        raise RuntimeError("Qwen OOF labels do not match labels file -- check row order.")

    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))
    family_index = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)
    class_weights = np.asarray(postprocess["training_class_weights"], dtype=np.float64)
    final_logits = (
        action_logits / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"]) * family_logits[:, family_index]
        - float(postprocess["prior_beta"]) * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )
    qwen_proba = stable_softmax(final_logits)
    sorted_proba = np.sort(qwen_proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]

    print("\nBuild file/search specialist OOF (this trains 5 LogisticRegression models, TF-IDF included)...")
    specialist_full_proba = build_specialist_oof(
        samples, all_labels, args.word_features, args.char_features, args.n_splits, args.seed,
    )
    np.save(args.output_dir / "specialist_full_oof_proba.npy", specialist_full_proba)

    qwen_pred = qwen_proba.argmax(axis=1)
    qwen_f1 = macro_f1(all_labels, qwen_pred)
    print(f"\nQwen baseline Macro-F1: {qwen_f1:.6f}")

    print("\n=== Full-data reference sweep (NOT the trust metric) ===")
    top1_in_group = np.isin(qwen_pred, GROUP_IDS)
    best_full = None
    for m in args.margin_grid:
        eligible = top1_in_group & (margin < m)
        for w in args.weight_grid:
            pred = blend_predict(qwen_proba, specialist_full_proba, eligible, w)
            f1 = macro_f1(all_labels, pred)
            gain = f1 - qwen_f1
            if best_full is None or f1 > best_full[0]:
                best_full = (f1, m, w, int(eligible.sum()))
    print(f"Best on full data: margin<{best_full[1]}, weight={best_full[2]}, "
          f"eligible={best_full[3]}, f1={best_full[0]:.6f}, gain={best_full[0]-qwen_f1:+.6f}")

    print(f"\n=== Nested CV ({args.n_folds} folds x {args.n_seeds} seeds = {args.n_folds*args.n_seeds} evals) ===")
    cv_results = nested_cv_screen(
        all_labels, qwen_proba, specialist_full_proba, margin,
        args.margin_grid, args.weight_grid, args.n_folds, args.n_seeds,
    )
    print(json.dumps(cv_results, indent=2))

    positive_rate = cv_results["nested_cv_positive"] / cv_results["nested_cv_total"]
    if positive_rate >= 0.8 and cv_results["nested_cv_gain_mean"] > 0.0005:
        verdict = "PROMISING - worth ONE careful public submission (remember V30: even this is not a guarantee)"
    elif positive_rate >= 0.5:
        verdict = "MARGINAL - not worth the submission risk given V30's precedent"
    else:
        verdict = "DISCARD"
    print(f"\nVerdict: {verdict}")

    if args.report:
        results = {
            "qwen_f1": qwen_f1,
            "full_data_best": {"margin": best_full[1], "weight": best_full[2], "eligible": best_full[3], "f1": best_full[0], "gain": best_full[0]-qwen_f1},
            **cv_results,
            "verdict": verdict,
        }
        (args.output_dir / "file_search_specialist_screen_report.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nSaved report to {args.output_dir}")


if __name__ == "__main__":
    main()
