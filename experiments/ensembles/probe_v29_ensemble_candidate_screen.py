"""
probe_v29_ensemble_candidate_screen.py

Fast (CPU, minutes not hours) screening tool for new ensemble-partner
candidates, BEFORE committing to a full blend probe (like probe_v23b) or
any GPU training.

Why this exists: E5 looked complementary on paper too (Qwen wrong/E5
correct=466, Qwen correct/E5 wrong=968, ratio 0.48 -- actually a BETTER
ratio than tree's 0.26) but didn't help in blend, while tree (worse ratio)
did. The likely reason: E5 is another fine-tuned neural LM, so its error
pattern correlates with Qwen's in ways raw disagreement counts don't
capture. Tree's TF-IDF/regex feature basis is a genuinely different signal
source. So "does this candidate disagree with Qwen in scale" matters less
than "is this candidate's failure mode structurally different from Qwen's".

This script trains a candidate classifier with leak-free 5-fold OOF and
reports the same diagnostics used earlier in the project (oracle gain,
disagreement counts, ratio) PLUS a quick fixed-weight log-prob blend check
with the same multi-seed holdout discipline as probe_v23b, so you get a
go/no-go signal in minutes instead of hours.

Two candidate modes:
  --features structured   -> LightGBM on the 95-dim structured features only
                              (no TF-IDF at all -- pure state-transition signal)
  --features tfidf         -> same TF-IDF+structured feature space V23's tree
                              used, but you can swap --model catboost to test
                              algorithmic diversity instead of feature diversity
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils_qwen_v4 import ALL_CLASSES, build_structured_features, build_segments

NUM_CLASSES = len(ALL_CLASSES)
LABEL2ID = {label: i for i, label in enumerate(ALL_CLASSES)}


def load_jsonl(path: Path) -> List[dict]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def macro_f1(labels, predictions) -> float:
    return f1_score(labels, predictions, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0)


def build_text_for_tfidf(sample: dict) -> str:
    segments = build_segments(sample)
    return " ".join([segments["current"], segments["action"], segments["history"], segments["meta"]])


def get_model(name: str, seed: int):
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=200, num_leaves=31, max_depth=8, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.7, min_child_samples=20,
            random_state=seed, n_jobs=-1, verbosity=-1,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.08, loss_function="MultiClass",
            random_seed=seed, verbose=False, thread_count=-1,
        )
    raise ValueError(f"unknown model: {name}")


def oof_predict_proba(
    X_all,
    labels: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    n_splits: int,
    seed: int,
    is_sparse: bool,
) -> np.ndarray:
    oof_proba = np.zeros((len(labels), NUM_CLASSES), dtype=np.float64)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups), start=1):
        X_train = X_all[train_idx]
        X_val = X_all[val_idx]
        model = get_model(model_name, seed)
        model.fit(X_train, labels[train_idx])
        local_proba = model.predict_proba(X_val)

        full_proba = np.full((len(val_idx), NUM_CLASSES), 1e-9)
        classes_seen = np.asarray(model.classes_).astype(int)
        full_proba[:, classes_seen] = local_proba
        full_proba /= full_proba.sum(axis=1, keepdims=True)
        oof_proba[val_idx] = full_proba

        fold_f1 = macro_f1(labels[val_idx], full_proba.argmax(axis=1))
        print(f"  Fold {fold}/{n_splits}: macro_f1={fold_f1:.6f}")

    return oof_proba


def disagreement_report(qwen_pred, qwen_proba, cand_pred, cand_proba, labels) -> dict:
    qwen_correct = qwen_pred == labels
    cand_correct = cand_pred == labels

    qwen_wrong_cand_correct = int(((~qwen_correct) & cand_correct).sum())
    qwen_correct_cand_wrong = int((qwen_correct & (~cand_correct)).sum())

    oracle_pred = np.where(qwen_correct, qwen_pred, np.where(cand_correct, cand_pred, qwen_pred))
    oracle_f1 = macro_f1(labels, oracle_pred)
    qwen_f1 = macro_f1(labels, qwen_pred)
    cand_f1 = macro_f1(labels, cand_pred)

    ratio = qwen_wrong_cand_correct / max(1, qwen_correct_cand_wrong)

    return {
        "qwen_f1": qwen_f1,
        "candidate_f1": cand_f1,
        "qwen_wrong_candidate_correct": qwen_wrong_cand_correct,
        "qwen_correct_candidate_wrong": qwen_correct_cand_wrong,
        "complementarity_ratio": ratio,
        "oracle_f1": oracle_f1,
        "oracle_gain": oracle_f1 - qwen_f1,
    }


def holdout_blend_check(qwen_proba, cand_proba, labels, weight: float, n_seeds: int, holdout_frac: float) -> dict:
    n = len(labels)
    rng_master = np.random.default_rng(0)
    gains = []
    for _ in range(n_seeds):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        idx = rng.choice(n, size=int(n * holdout_frac), replace=False)
        qwen_pred = qwen_proba[idx].argmax(axis=1)
        qwen_f1 = macro_f1(labels[idx], qwen_pred)

        log_p = (1 - weight) * np.log(np.maximum(qwen_proba[idx], 1e-12)) + weight * np.log(np.maximum(cand_proba[idx], 1e-12))
        blend_pred = log_p.argmax(axis=1)
        blend_f1 = macro_f1(labels[idx], blend_pred)
        gains.append(blend_f1 - qwen_f1)

    gains = np.asarray(gains)
    return {
        "blend_weight": weight,
        "holdout_gain_mean": float(gains.mean()),
        "holdout_gain_min": float(gains.min()),
        "holdout_positive": int((gains > 0).sum()),
        "holdout_total": n_seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--qwen-oof", type=Path, required=True, help="model/qwen_v4_oof/oof_logits_all_70000.npz")
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--baseline-tree-oof-proba", type=Path, default=None,
                         help="if given, screens the candidate against qwen+tree (at --baseline-tree-weight) "
                              "instead of qwen alone -- use this once tree is already locked into the blend, "
                              "since a candidate that only helps vs qwen alone may add nothing once tree is present")
    parser.add_argument("--baseline-tree-weight", type=float, default=0.25)
    parser.add_argument("--features", choices=["structured", "tfidf"], default="structured")
    parser.add_argument("--model", choices=["lightgbm", "catboost"], default="lightgbm")
    parser.add_argument("--word-features", type=int, default=16000, help="only used with --features tfidf")
    parser.add_argument("--char-features", type=int, default=16000, help="only used with --features tfidf")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blend-weight", type=float, default=0.25)
    parser.add_argument("--sweep-blend-weight", action="store_true",
                         help="try a range of weights and report the best on full data before the holdout check")
    parser.add_argument("--holdout-seeds", type=int, default=10)
    parser.add_argument("--holdout-frac", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Load data...")
    samples = load_jsonl(args.data)
    with args.labels_csv.open(encoding="utf-8", newline="") as f:
        label_map = {str(row["id"]): str(row["action"]) for row in csv.DictReader(f)}
    labels = np.asarray([LABEL2ID[label_map[str(s["id"])]] for s in samples], dtype=np.int64)
    groups = np.asarray([str(s["id"]).rsplit("-step_", 1)[0] for s in samples])
    print(f"Rows: {len(samples)} classes={NUM_CLASSES} groups={len(set(groups))}")

    print(f"Build candidate features (mode={args.features})...")
    if args.features == "structured":
        X_all = np.stack([build_structured_features(s) for s in samples]).astype(np.float32)
        is_sparse = False
        print(f"Structured feature matrix: {X_all.shape}")
    else:
        texts = [build_text_for_tfidf(s) for s in samples]
        word_vec = TfidfVectorizer(max_features=args.word_features, ngram_range=(1, 2), min_df=2, max_df=0.98)
        char_vec = TfidfVectorizer(max_features=args.char_features, analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.98)
        word_X = word_vec.fit_transform(texts)
        char_X = char_vec.fit_transform(texts)
        structured_X = sparse.csr_matrix(np.stack([build_structured_features(s) for s in samples]).astype(np.float32))
        X_all = sparse.hstack([word_X, char_X, structured_X]).tocsr()
        is_sparse = True
        print(f"TF-IDF+structured feature matrix: {X_all.shape}")

    print("Load Qwen OOF...")
    oof = np.load(args.qwen_oof)
    qwen_action = oof["action_logits"].astype(np.float64)
    qwen_family = oof["family_logits"].astype(np.float64)
    oof_labels = oof["labels"].astype(np.int64)
    if not np.array_equal(labels, oof_labels):
        raise RuntimeError("Qwen OOF labels do not match labels file -- check row order / --data matches what OOF was generated from.")

    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))
    from feature_utils_qwen_v4 import ACTION_TO_FAMILY
    family_index = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)
    class_weights = np.asarray(postprocess["training_class_weights"], dtype=np.float64)
    qwen_final = (
        qwen_action / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"]) * qwen_family[:, family_index]
        - float(postprocess["prior_beta"]) * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )
    qwen_proba = stable_softmax(qwen_final)
    qwen_pred = qwen_proba.argmax(axis=1)

    if args.baseline_tree_oof_proba is not None:
        tree_proba = np.load(args.baseline_tree_oof_proba)
        assert tree_proba.shape[0] == len(labels), (
            f"tree_oof rows={tree_proba.shape[0]} vs labels={len(labels)} -- "
            "confirm it was generated on the same --data row order"
        )
        w = args.baseline_tree_weight
        baseline_log_p = (1 - w) * np.log(np.maximum(qwen_proba, 1e-12)) + w * np.log(np.maximum(tree_proba, 1e-12))
        baseline_proba = stable_softmax(baseline_log_p)
        baseline_f1 = macro_f1(labels, baseline_proba.argmax(axis=1))
        print(f"Screening against qwen+tree (weight={w}), baseline_f1={baseline_f1:.6f} "
              f"(vs qwen alone {macro_f1(labels, qwen_pred):.6f})")
    else:
        baseline_proba = qwen_proba
        print("Screening against qwen alone (pass --baseline-tree-oof-proba to screen against qwen+tree instead)")

    baseline_pred = baseline_proba.argmax(axis=1)

    print(f"Train candidate ({args.model}) with {args.n_splits}-fold leak-free OOF...")
    cand_proba = oof_predict_proba(X_all, labels, groups, args.model, args.n_splits, args.seed, is_sparse)
    cand_pred = cand_proba.argmax(axis=1)

    print()
    print("=== Disagreement / oracle report ===")
    disagreement = disagreement_report(baseline_pred, baseline_proba, cand_pred, cand_proba, labels)
    print(json.dumps(disagreement, indent=2))

    print()
    print(f"=== Fixed-weight blend check ===")
    if args.sweep_blend_weight:
        baseline_f1_all = macro_f1(labels, baseline_pred)
        best_weight, best_gain = args.blend_weight, -1.0
        for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]:
            log_p = (1 - w) * np.log(np.maximum(baseline_proba, 1e-12)) + w * np.log(np.maximum(cand_proba, 1e-12))
            pred = log_p.argmax(axis=1)
            f1 = macro_f1(labels, pred)
            gain = f1 - baseline_f1_all
            print(f"  weight={w:.2f} full_f1={f1:.6f} gain_vs_baseline={gain:+.6f}")
            if gain > best_gain:
                best_gain = gain
                best_weight = w
        print(f"Best weight on full data (reference only, not the trust metric): {best_weight}")
        args.blend_weight = best_weight

    print(f"({args.holdout_seeds}x holdout, weight={args.blend_weight})")
    blend = holdout_blend_check(baseline_proba, cand_proba, labels, args.blend_weight, args.holdout_seeds, args.holdout_frac)
    print(json.dumps(blend, indent=2))

    positive_rate = blend["holdout_positive"] / blend["holdout_total"]
    if positive_rate >= 0.8 and blend["holdout_gain_mean"] > 0.0005:
        verdict = "PROMISING - worth a full probe_v23b-style weight sweep + full-data OOF for a real submission candidate"
    elif positive_rate >= 0.5:
        verdict = "MARGINAL - some signal but not decisively better than V23 tree; try tuning weight or feature set before investing more"
    else:
        verdict = "DISCARD - same failure pattern as E5: doesn't complement Qwen's errors on holdout"
    print(f"\nVerdict: {verdict}")

    results = {"disagreement": disagreement, "blend_check": blend, "verdict": verdict,
               "features": args.features, "model": args.model}
    if args.report:
        (args.output_dir / "v29_screen_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        np.save(args.output_dir / "candidate_oof_proba.npy", cand_proba)
        print(f"Saved report + OOF proba to {args.output_dir}")


if __name__ == "__main__":
    main()
