#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train and evaluate a compact 3-way specialist for:
- run_bash
- run_tests
- lint_or_typecheck

The specialist is trained only on fold-train rows whose true label belongs to
the exec/check family. During validation it is allowed to alter predictions
only when the V12 top-1 prediction is also in that family.

Example:
python .\train_exec_check_specialist.py `
  --data .\data\train.jsonl `
  --labels .\data\train_labels.csv `
  --v12-logits .\model\qwen_distill_v12_eval\validation_logits_v12.npz `
  --output-dir .\model\exec_check_specialist_eval
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


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
LABEL2ID = {name: i for i, name in enumerate(CLASSES)}

EXEC_CLASSES = ["run_bash", "run_tests", "lint_or_typecheck"]
EXEC_GLOBAL_IDS = np.asarray([LABEL2ID[name] for name in EXEC_CLASSES], dtype=np.int64)
GLOBAL_TO_LOCAL = {global_id: local_id for local_id, global_id in enumerate(EXEC_GLOBAL_IDS)}
LOCAL_TO_GLOBAL = {local_id: global_id for global_id, local_id in GLOBAL_TO_LOCAL.items()}

ACTION_TO_FAMILY = np.asarray(
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 3],
    dtype=np.int64,
)

V12_CLASS_WEIGHTS = np.asarray(
    [
        0.8572690251352013,
        0.84274223991845,
        1.0366924510397455,
        0.9862908339428644,
        0.8179509647354836,
        1.3554499048325421,
        1.009072807992434,
        0.996587290637956,
        1.0232914644762283,
        1.2165721371457607,
        1.166404985555288,
        1.1687102789899324,
        1.407913957899642,
        0.9913123924717526,
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--v12-logits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--eval-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--word-features", type=int, default=20000)
    parser.add_argument("--char-features", type=int, default=20000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=1200)
    parser.add_argument("--c", type=float, default=3.0)

    parser.add_argument("--v12-action-temperature", type=float, default=0.6)
    parser.add_argument("--v12-family-weight", type=float, default=0.15)
    parser.add_argument("--v12-prior-beta", type=float, default=0.25)

    parser.add_argument(
        "--specialist-weights",
        type=str,
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument(
        "--min-confidences",
        type=str,
        default="0,0.4,0.5,0.6,0.7,0.8",
    )
    parser.add_argument(
        "--max-v12-margins",
        type=str,
        default="1.0,0.5,0.3,0.2,0.1,0.05",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_targets(path: Path, samples: List[dict]) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as file:
        mapping = {str(row["id"]): str(row["action"]) for row in csv.DictReader(file)}
    targets = []
    for sample in samples:
        sample_id = str(sample["id"])
        action = mapping.get(sample_id)
        if action not in LABEL2ID:
            raise RuntimeError(f"Missing or invalid label for id={sample_id}: {action}")
        targets.append(LABEL2ID[action])
    return np.asarray(targets, dtype=np.int64)


def stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_text(sample: dict) -> str:
    current = stringify(
        sample.get("current_prompt")
        or sample.get("prompt")
        or sample.get("instruction")
        or sample.get("query")
    )
    history = stringify(
        sample.get("history")
        or sample.get("messages")
        or sample.get("conversation")
    )
    workspace = stringify(sample.get("workspace"))
    meta = stringify(sample.get("session_meta"))

    # Keep the current prompt duplicated because it is usually the strongest
    # feature, while retaining history/tool traces that often determine the label.
    return (
        f"[CURRENT] {current}\n"
        f"[CURRENT_REPEAT] {current}\n"
        f"[HISTORY] {history[-5000:]}\n"
        f"[WORKSPACE] {workspace[-1500:]}\n"
        f"[META] {meta[-1000:]}"
    )


def stable_softmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x -= x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-300)


def compute_v12_probability(payload: Dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    final = payload["action_logits"].astype(np.float64) / args.v12_action_temperature
    final += (
        args.v12_family_weight
        * payload["family_logits"].astype(np.float64)[:, ACTION_TO_FAMILY]
    )
    final -= (
        args.v12_prior_beta
        * np.log(np.maximum(V12_CLASS_WEIGHTS, 1e-12))[None, :]
    )
    return stable_softmax(final)


def resolve_fold(
    samples: List[dict],
    labels: np.ndarray,
    eval_fold: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    groups = np.asarray([str(s["id"]).rsplit("-step_", 1)[0] for s in samples])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups)
    ):
        if fold == eval_fold:
            return train_idx.astype(np.int64), val_idx.astype(np.int64)
    raise ValueError(f"Invalid eval_fold={eval_fold}")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(len(CLASSES))),
            average="macro",
            zero_division=0,
        )
    )


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Load data...")
    samples = load_jsonl(args.data)
    labels = load_targets(args.labels, samples)
    train_idx, val_idx = resolve_fold(samples, labels, args.eval_fold, args.seed)

    with np.load(args.v12_logits) as loaded:
        v12_payload = {key: loaded[key] for key in loaded.files}

    if v12_payload["labels"].shape[0] != len(val_idx):
        raise RuntimeError(
            f"V12 rows={v12_payload['labels'].shape[0]} but fold rows={len(val_idx)}"
        )
    if not np.array_equal(v12_payload["labels"].astype(np.int64), labels[val_idx]):
        raise RuntimeError("V12 labels do not align with reconstructed fold.")

    texts = [build_text(sample) for sample in samples]

    train_exec_mask = np.isin(labels[train_idx], EXEC_GLOBAL_IDS)
    train_exec_idx = train_idx[train_exec_mask]
    train_exec_y = np.asarray(
        [GLOBAL_TO_LOCAL[int(value)] for value in labels[train_exec_idx]],
        dtype=np.int64,
    )

    val_exec_true_mask = np.isin(labels[val_idx], EXEC_GLOBAL_IDS)
    print("Fold train rows:", len(train_idx))
    print("Specialist train rows:", len(train_exec_idx))
    print("Validation rows:", len(val_idx))
    print("Validation true exec/check rows:", int(val_exec_true_mask.sum()))

    train_texts = [texts[int(i)] for i in train_exec_idx]
    val_texts = [texts[int(i)] for i in val_idx]

    print("Fit word TF-IDF...")
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_features=args.word_features,
        sublinear_tf=True,
        strip_accents=None,
        lowercase=True,
        dtype=np.float32,
    )
    x_word_train = word.fit_transform(train_texts)
    x_word_val = word.transform(val_texts)

    print("Fit char TF-IDF...")
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=args.min_df,
        max_features=args.char_features,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )
    x_char_train = char.fit_transform(train_texts)
    x_char_val = char.transform(val_texts)

    x_train = sparse.hstack([x_word_train, x_char_train], format="csr")
    x_val = sparse.hstack([x_word_val, x_char_val], format="csr")
    print("Feature shape:", x_train.shape)

    classifier = LogisticRegression(
        C=args.c,
        max_iter=args.max_iter,
        class_weight="balanced",
        solver="liblinear",
        random_state=args.seed,
    )
    print("Fit specialist...")
    classifier.fit(x_train, train_exec_y)

    specialist_probability = classifier.predict_proba(x_val).astype(np.float64)
    specialist_prediction_local = specialist_probability.argmax(axis=1)
    specialist_prediction_global = np.asarray(
        [LOCAL_TO_GLOBAL[int(value)] for value in specialist_prediction_local],
        dtype=np.int64,
    )

    v12_probability = compute_v12_probability(v12_payload, args)
    v12_prediction = v12_probability.argmax(axis=1)
    y_val = labels[val_idx]
    base_score = macro_f1(y_val, v12_prediction)

    v12_exec_probability = v12_probability[:, EXEC_GLOBAL_IDS]
    v12_exec_probability /= np.maximum(
        v12_exec_probability.sum(axis=1, keepdims=True),
        1e-12,
    )
    v12_exec_prediction_local = v12_exec_probability.argmax(axis=1)
    v12_exec_margin = (
        np.sort(v12_exec_probability, axis=1)[:, -1]
        - np.sort(v12_exec_probability, axis=1)[:, -2]
    )

    weights = parse_float_list(args.specialist_weights)
    min_confidences = parse_float_list(args.min_confidences)
    max_margins = parse_float_list(args.max_v12_margins)

    rows = []
    predictions = {}

    for weight in weights:
        blended_exec = stable_softmax(
            (1.0 - weight) * np.log(np.maximum(v12_exec_probability, 1e-12))
            + weight * np.log(np.maximum(specialist_probability, 1e-12))
        )
        blended_local = blended_exec.argmax(axis=1)
        blended_global = np.asarray(
            [LOCAL_TO_GLOBAL[int(value)] for value in blended_local],
            dtype=np.int64,
        )

        for min_confidence in min_confidences:
            for max_margin in max_margins:
                eligible = (
                    np.isin(v12_prediction, EXEC_GLOBAL_IDS)
                    & (specialist_probability.max(axis=1) >= min_confidence)
                    & (v12_exec_margin <= max_margin)
                )
                candidate = v12_prediction.copy()
                candidate[eligible] = blended_global[eligible]
                score = macro_f1(y_val, candidate)

                key = (weight, min_confidence, max_margin)
                predictions[key] = candidate
                rows.append(
                    {
                        "specialist_weight": weight,
                        "min_specialist_confidence": min_confidence,
                        "max_v12_exec_margin": max_margin,
                        "macro_f1": score,
                        "gain_vs_v12": score - base_score,
                        "eligible_rows": int(eligible.sum()),
                        "changed_rows": int(np.sum(candidate != v12_prediction)),
                        "net_correct_changes": int(
                            np.sum((candidate == y_val) & (v12_prediction != y_val))
                            - np.sum((candidate != y_val) & (v12_prediction == y_val))
                        ),
                    }
                )

    results = pd.DataFrame(rows).sort_values(
        ["macro_f1", "changed_rows"],
        ascending=[False, True],
    )
    best = results.iloc[0]
    best_key = (
        float(best["specialist_weight"]),
        float(best["min_specialist_confidence"]),
        float(best["max_v12_exec_margin"]),
    )
    best_prediction = predictions[best_key]

    print("\n=== Baseline ===")
    print(f"V12 Macro-F1: {base_score:.9f}")

    print("\n=== Top specialist configurations ===")
    print(results.head(30).to_string(index=False))

    # Standalone specialist accuracy only on true exec/check validation samples.
    true_exec_local = np.asarray(
        [GLOBAL_TO_LOCAL[int(value)] for value in y_val[val_exec_true_mask]],
        dtype=np.int64,
    )
    specialist_true_exec_accuracy = float(
        np.mean(
            specialist_prediction_local[val_exec_true_mask]
            == true_exec_local
        )
    )

    summary = {
        "v12_macro_f1": base_score,
        "specialist_train_rows": int(len(train_exec_idx)),
        "validation_rows": int(len(val_idx)),
        "validation_true_exec_rows": int(val_exec_true_mask.sum()),
        "specialist_true_exec_accuracy": specialist_true_exec_accuracy,
        "best": {
            "specialist_weight": best_key[0],
            "min_specialist_confidence": best_key[1],
            "max_v12_exec_margin": best_key[2],
            "macro_f1": float(best["macro_f1"]),
            "gain_vs_v12": float(best["gain_vs_v12"]),
            "eligible_rows": int(best["eligible_rows"]),
            "changed_rows": int(best["changed_rows"]),
            "net_correct_changes": int(best["net_correct_changes"]),
        },
    }

    results.to_csv(
        args.output_dir / "specialist_sweep.csv",
        index=False,
        encoding="utf-8-sig",
    )

    changes = pd.DataFrame(
        {
            "global_row_index": val_idx,
            "id": [str(samples[int(i)]["id"]) for i in val_idx],
            "true_action": [CLASSES[int(i)] for i in y_val],
            "v12_pred": [CLASSES[int(i)] for i in v12_prediction],
            "specialist_pred": [CLASSES[int(i)] for i in specialist_prediction_global],
            "best_pred": [CLASSES[int(i)] for i in best_prediction],
            "changed": best_prediction != v12_prediction,
            "v12_exec_margin": v12_exec_margin,
            "specialist_confidence": specialist_probability.max(axis=1),
            "current_prompt": [
                stringify(
                    samples[int(i)].get("current_prompt")
                    or samples[int(i)].get("prompt")
                    or samples[int(i)].get("instruction")
                    or samples[int(i)].get("query")
                )
                for i in val_idx
            ],
        }
    )
    changes[changes["changed"]].to_csv(
        args.output_dir / "best_changed_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(
        {
            "word_vectorizer": word,
            "char_vectorizer": char,
            "classifier": classifier,
            "exec_classes": EXEC_CLASSES,
            "classes": CLASSES,
            "best_config": summary["best"],
        },
        args.output_dir / "exec_check_specialist.joblib",
        compress=3,
    )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
