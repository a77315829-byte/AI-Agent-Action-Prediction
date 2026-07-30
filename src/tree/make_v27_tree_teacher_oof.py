from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover
    raise RuntimeError("lightgbm is required for this script") from exc


def load_v23_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("v23_submission_module", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import V23 script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labels(path: Path, samples: Sequence[dict], label2id: Dict[str, int]) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as f:
        label_map = {str(row["id"]): label2id[str(row["action"])] for row in csv.DictReader(f)}
    return np.asarray([label_map[str(sample["id"])] for sample in samples], dtype=np.int64)


def session_groups(samples: Sequence[dict]) -> np.ndarray:
    return np.asarray([str(sample.get("id", "")).rsplit("-step_", 1)[0] for sample in samples])


def build_texts(v23, samples: Sequence[dict]) -> Tuple[List[str], List[str]]:
    texts = [v23.extract_text(sample) for sample in samples]
    current_texts = [v23.extract_current_only_text(sample) for sample in samples]
    return texts, current_texts


def build_features(v23, vectorizers: Dict[str, Any], samples: Sequence[dict], texts: List[str], current_texts: List[str], fit: bool):
    if fit:
        x_word = vectorizers["word"].fit_transform(texts)
        x_char = vectorizers["char"].fit_transform(current_texts)
        structured_dense = v23.structured_tree_features(samples, texts, current_texts)
        x_structured = sparse.csr_matrix(vectorizers["scaler"].fit_transform(structured_dense))
    else:
        x_word = vectorizers["word"].transform(texts)
        x_char = vectorizers["char"].transform(current_texts)
        structured_dense = v23.structured_tree_features(samples, texts, current_texts)
        x_structured = sparse.csr_matrix(vectorizers["scaler"].transform(structured_dense))
    return sparse.hstack([x_word, x_char, x_structured], format="csr")


def make_vectorizers(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "word": TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, args.word_ngram_max),
            min_df=args.min_df,
            max_df=args.max_df,
            max_features=args.word_features,
            sublinear_tf=True,
            strip_accents=None,
            lowercase=True,
        ),
        "char": TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(args.char_ngram_min, args.char_ngram_max),
            min_df=args.min_df,
            max_df=args.max_df,
            max_features=args.char_features,
            sublinear_tf=True,
            lowercase=True,
        ),
        "scaler": StandardScaler(),
    }


def make_lgbm(args: argparse.Namespace, num_classes: int, seed: int):
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=seed,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )


def aligned_predict_proba(model, x, num_classes: int) -> np.ndarray:
    local = model.predict_proba(x)
    result = np.full((x.shape[0], num_classes), 1e-12, dtype=np.float64)
    classes = getattr(model, "classes_", np.arange(local.shape[1])).astype(int)
    result[:, classes] = local
    result /= result.sum(axis=1, keepdims=True)
    return result


def macro_f1(labels: np.ndarray, pred: np.ndarray, num_classes: int) -> float:
    return float(f1_score(labels, pred, labels=np.arange(num_classes), average="macro", zero_division=0))


def save_bundle(path: Path, vectorizers: Dict[str, Any], model: Any, config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "word_vectorizer": vectorizers["word"],
            "char_vectorizer": vectorizers["char"],
            "scaler": vectorizers["scaler"],
            "model": model,
            "config": config,
        },
        path,
        compress=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--tree-script", type=Path, default=Path("script_submission_v23_qwen_tree_blend.py"))
    parser.add_argument("--output", type=Path, default=Path("model/v27_tree_teacher_oof/tree_prob_oof_all.npy"))
    parser.add_argument("--metadata-output", type=Path, default=Path("model/v27_tree_teacher_oof/metadata.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("model/v27_tree_teacher_oof/fold_artifacts"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--word-features", type=int, default=30000)
    parser.add_argument("--char-features", type=int, default=30000)
    parser.add_argument("--word-ngram-max", type=int, default=2)
    parser.add_argument("--char-ngram-min", type=int, default=3)
    parser.add_argument("--char-ngram-max", type=int, default=5)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.98)
    parser.add_argument("--n-estimators", type=int, default=360)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=30)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.60)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--save-fold-artifacts", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    v23 = load_v23_module(args.tree_script)
    classes = list(v23.ALL_CLASSES)
    label2id = {name: i for i, name in enumerate(classes)}
    num_classes = len(classes)

    print(f"Load data: {args.data}")
    samples = load_jsonl(args.data)
    labels = load_labels(args.labels_csv, samples, label2id)
    groups = session_groups(samples)
    print(f"Rows: {len(samples)} classes={num_classes} groups={len(set(groups))}")

    print("Build all texts once...")
    texts, current_texts = build_texts(v23, samples)

    oof = np.full((len(samples), num_classes), 1e-12, dtype=np.float64)
    fold_scores: List[Dict[str, Any]] = []

    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups), start=1):
        print(f"\nFold {fold}/{args.n_splits}: train={len(train_idx)} val={len(val_idx)}")
        train_samples = [samples[int(i)] for i in train_idx]
        val_samples = [samples[int(i)] for i in val_idx]
        train_texts = [texts[int(i)] for i in train_idx]
        val_texts = [texts[int(i)] for i in val_idx]
        train_current = [current_texts[int(i)] for i in train_idx]
        val_current = [current_texts[int(i)] for i in val_idx]

        vectorizers = make_vectorizers(args)
        print("  Fit/transform features...")
        x_train = build_features(v23, vectorizers, train_samples, train_texts, train_current, fit=True)
        x_val = build_features(v23, vectorizers, val_samples, val_texts, val_current, fit=False)
        print(f"  x_train={x_train.shape} x_val={x_val.shape}")

        model = make_lgbm(args, num_classes=num_classes, seed=args.seed + fold)
        print("  Fit LightGBM...")
        model.fit(x_train, labels[train_idx])
        prob = aligned_predict_proba(model, x_val, num_classes=num_classes)
        oof[val_idx] = prob
        pred = prob.argmax(axis=1)
        score = macro_f1(labels[val_idx], pred, num_classes)
        fold_scores.append({"fold": fold, "macro_f1": score, "train": int(len(train_idx)), "val": int(len(val_idx))})
        print(f"  Fold Macro-F1: {score:.6f}")

        if args.save_fold_artifacts:
            save_bundle(
                args.artifact_dir / f"fold{fold}.joblib",
                vectorizers,
                model,
                config=vars(args) | {"fold": fold, "classes": classes},
            )

    oof /= oof.sum(axis=1, keepdims=True)
    oof_pred = oof.argmax(axis=1)
    oof_score = macro_f1(labels, oof_pred, num_classes)
    print(f"\nOOF Tree Macro-F1: {oof_score:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, oof.astype(np.float32))
    print(f"Saved: {args.output} {oof.shape}")

    metadata = {
        "rows": int(len(samples)),
        "classes": classes,
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "oof_tree_macro_f1": float(oof_score),
        "fold_scores": fold_scores,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "note": "Leak-free tree OOF probabilities. Use this for V27 distillation; do not use full-train tree_prob_all for local validation.",
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metadata: {args.metadata_output}")

    if args.report:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
