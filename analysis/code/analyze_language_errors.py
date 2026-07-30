#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze whether errors are concentrated in Korean / English / mixed-language samples.

Run from project root, for example:

python .\analyze_language_errors.py `
  --data .\data\train.jsonl `
  --labels .\data\train_labels.csv `
  --logits .\model\qwen_distill_v12_eval\validation_logits_v12.npz `
  --postprocess .\model\qwen_distill_v12_eval\postprocess.json `
  --output-dir .\model\lang_error_v12

Optional, to analyze Qwen+tree blend instead of Qwen-only:
python .\analyze_language_errors.py `
  --data .\data\train.jsonl `
  --labels .\data\train_labels.csv `
  --logits .\model\qwen_distill_v12_eval\validation_logits_v12.npz `
  --postprocess .\model\qwen_distill_v12_eval\postprocess.json `
  --tree-proba .\model\v23_tree_lgbm_full\tree_prob_val.npy `
  --tree-weight 0.25 `
  --output-dir .\model\lang_error_v23_blend
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def get_current_text(sample: Dict[str, Any]) -> str:
    for key in ["current_prompt", "prompt", "instruction", "query"]:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value

    try:
        from feature_utils_qwen_v4 import build_segments
        segments = build_segments(sample)
        value = segments.get("current", "")
        if isinstance(value, str) and value.strip():
            return value
    except Exception:
        pass

    return json.dumps(sample, ensure_ascii=False)


_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def language_bucket(text: str) -> Tuple[str, Dict[str, float]]:
    hangul = len(_HANGUL_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    total_letters = hangul + latin

    ko_ratio = hangul / max(total_letters, 1)
    latin_ratio = latin / max(total_letters, 1)

    if hangul >= 3 and latin >= 10:
        bucket = "mixed"
    elif hangul >= 3:
        bucket = "ko"
    elif latin >= 10:
        bucket = "en"
    elif hangul > 0 and latin > 0:
        bucket = "mixed"
    elif hangul > 0:
        bucket = "ko"
    elif latin > 0:
        bucket = "en"
    else:
        bucket = "other"

    return bucket, {
        "hangul_chars": float(hangul),
        "latin_chars": float(latin),
        "ko_ratio": float(ko_ratio),
        "latin_ratio": float(latin_ratio),
    }


def get_class_info() -> Tuple[List[str], np.ndarray]:
    try:
        from feature_utils_qwen_v4 import ALL_CLASSES, ACTION_TO_FAMILY
        return list(ALL_CLASSES), np.asarray(ACTION_TO_FAMILY, dtype=np.int64)
    except Exception as e:
        raise RuntimeError(
            "Could not import ALL_CLASSES / ACTION_TO_FAMILY from feature_utils_qwen_v4.py. "
            "Run this script from project root or place feature_utils_qwen_v4.py next to it."
        ) from e


def read_labels(labels_csv: Path, classes: List[str], samples: List[dict]) -> np.ndarray:
    label2id = {c: i for i, c in enumerate(classes)}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        label_map = {str(row["id"]): str(row["action"]) for row in csv.DictReader(f)}
    labels = []
    for s in samples:
        sid = str(s["id"])
        if sid not in label_map:
            raise RuntimeError(f"Missing label for sample id={sid}")
        labels.append(label2id[label_map[sid]])
    return np.asarray(labels, dtype=np.int64)


def resolve_validation_indices(samples: List[dict], labels: np.ndarray, eval_fold: int, seed: int) -> np.ndarray:
    groups = np.asarray([str(s["id"]).rsplit("-step_", 1)[0] for s in samples])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        if fold == eval_fold:
            return np.asarray(val_idx, dtype=np.int64)
    raise RuntimeError(f"eval_fold {eval_fold} not found")


def load_logits(logits_path: Path) -> Dict[str, np.ndarray]:
    z = np.load(logits_path)
    return {k: z[k] for k in z.files}


def compute_qwen_proba(logits: Dict[str, np.ndarray], postprocess_path: Path, family_index: np.ndarray) -> np.ndarray:
    action_logits = logits["action_logits"].astype(np.float64)

    post = json.loads(postprocess_path.read_text(encoding="utf-8"))
    action_temperature = float(post.get("action_temperature", 1.0))
    family_weight = float(post.get("family_weight", 0.0))
    prior_beta = float(post.get("prior_beta", 0.0))

    final = action_logits / action_temperature

    if "family_logits" in logits and family_weight != 0.0:
        family_logits = logits["family_logits"].astype(np.float64)
        final = final + family_weight * family_logits[:, family_index]

    class_weights = post.get("training_class_weights")
    if class_weights is not None and prior_beta != 0.0:
        cw = np.asarray(class_weights, dtype=np.float64)
        final = final - prior_beta * np.log(np.maximum(cw, 1e-12))[None, :]

    return stable_softmax(final)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def summarize_by_language(df: pd.DataFrame, classes: List[str]) -> pd.DataFrame:
    rows = []
    all_label_ids = list(range(len(classes)))

    for bucket, part in df.groupby("lang"):
        y = part["label_id"].to_numpy(dtype=np.int64)
        p = part["pred_id"].to_numpy(dtype=np.int64)
        rows.append({
            "lang": bucket,
            "n": len(part),
            "accuracy": accuracy_score(y, p),
            "macro_f1_all_classes": macro_f1(y, p, all_label_ids),
            "macro_f1_present_classes": macro_f1(y, p, sorted(set(y.tolist()))),
            "error_rate": float((y != p).mean()),
            "wrong": int((y != p).sum()),
            "avg_margin": float(part["margin"].mean()),
            "median_margin": float(part["margin"].median()),
        })

    return pd.DataFrame(rows).sort_values("n", ascending=False)


def per_class_by_language(df: pd.DataFrame, classes: List[str]) -> pd.DataFrame:
    rows = []
    for bucket, part in df.groupby("lang"):
        y = part["label_id"].to_numpy(dtype=np.int64)
        p = part["pred_id"].to_numpy(dtype=np.int64)
        pr, rc, f1, support = precision_recall_fscore_support(
            y, p, labels=list(range(len(classes))), zero_division=0
        )
        for i, name in enumerate(classes):
            rows.append({
                "lang": bucket,
                "class": name,
                "support": int(support[i]),
                "precision": float(pr[i]),
                "recall": float(rc[i]),
                "f1": float(f1[i]),
                "wrong": int(((part["label_id"] == i) & (part["pred_id"] != i)).sum()),
            })
    return pd.DataFrame(rows).sort_values(["lang", "f1", "support"], ascending=[True, True, False])


def top_confusions_by_language(df: pd.DataFrame, classes: List[str], top_k: int) -> pd.DataFrame:
    rows = []
    wrong = df[df["label_id"] != df["pred_id"]]
    for bucket, part in wrong.groupby("lang"):
        counter = Counter(zip(part["label_id"], part["pred_id"]))
        for (true_id, pred_id), count in counter.most_common(top_k):
            rows.append({
                "lang": bucket,
                "true": classes[int(true_id)],
                "pred": classes[int(pred_id)],
                "count": int(count),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--labels", "--labels-csv", dest="labels_csv", type=Path, required=True)
    ap.add_argument("--logits", type=Path, required=True)
    ap.add_argument("--postprocess", type=Path, required=True)
    ap.add_argument("--eval-fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tree-proba", type=Path, default=None)
    ap.add_argument("--tree-weight", type=float, default=0.25)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--top-k-confusions", type=int, default=25)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    classes, family_index = get_class_info()
    samples = load_jsonl(args.data)
    labels_all = read_labels(args.labels_csv, classes, samples)
    val_idx = resolve_validation_indices(samples, labels_all, args.eval_fold, args.seed)

    logits = load_logits(args.logits)
    qwen_proba = compute_qwen_proba(logits, args.postprocess, family_index)

    if qwen_proba.shape[0] == len(samples):
        rows_idx = np.arange(len(samples))
        y = labels_all
        used_samples = samples
    elif qwen_proba.shape[0] == len(val_idx):
        rows_idx = val_idx
        y = labels_all[val_idx]
        used_samples = [samples[i] for i in val_idx]
    else:
        raise RuntimeError(
            f"Cannot align logits rows={qwen_proba.shape[0]} with full={len(samples)} or fold{args.eval_fold}={len(val_idx)}"
        )

    proba = qwen_proba
    model_name = "qwen"

    if args.tree_proba is not None:
        tree = np.load(args.tree_proba).astype(np.float64)
        if tree.shape[0] == len(samples) and proba.shape[0] == len(samples):
            tree_used = tree
        elif tree.shape[0] == len(val_idx) and proba.shape[0] == len(val_idx):
            tree_used = tree
        elif tree.shape[0] == len(samples) and proba.shape[0] == len(val_idx):
            tree_used = tree[val_idx]
        else:
            raise RuntimeError(f"Cannot align tree rows={tree.shape[0]} with proba rows={proba.shape[0]}")
        w = float(args.tree_weight)
        logp = (1.0 - w) * np.log(np.maximum(proba, 1e-12)) + w * np.log(np.maximum(tree_used, 1e-12))
        proba = stable_softmax(logp)
        model_name = f"qwen_tree_w{w}"

    pred = proba.argmax(axis=1)
    sorted_proba = np.sort(proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]

    records = []
    for local_i, sample in enumerate(used_samples):
        text = get_current_text(sample)
        bucket, lang_stats = language_bucket(text)
        records.append({
            "row_index": int(rows_idx[local_i]),
            "id": str(sample.get("id", "")),
            "lang": bucket,
            "label_id": int(y[local_i]),
            "label": classes[int(y[local_i])],
            "pred_id": int(pred[local_i]),
            "pred": classes[int(pred[local_i])],
            "correct": bool(pred[local_i] == y[local_i]),
            "confidence": float(proba[local_i, pred[local_i]]),
            "margin": float(margin[local_i]),
            **lang_stats,
            "current_text_preview": text[:300].replace("\n", " "),
        })

    df = pd.DataFrame(records)
    summary = summarize_by_language(df, classes)
    class_table = per_class_by_language(df, classes)
    confusions = top_confusions_by_language(df, classes, args.top_k_confusions)

    overall = {
        "model": model_name,
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(df["label_id"], df["pred_id"])),
        "macro_f1": macro_f1(df["label_id"].to_numpy(), df["pred_id"].to_numpy(), list(range(len(classes)))),
        "wrong": int((~df["correct"]).sum()),
        "language_counts": df["lang"].value_counts().to_dict(),
    }

    print("\n=== Overall ===")
    print(json.dumps(overall, ensure_ascii=False, indent=2))

    print("\n=== By language ===")
    print(summary.to_string(index=False))

    print("\n=== Top confusions by language ===")
    if len(confusions):
        print(confusions.head(args.top_k_confusions * max(1, df["lang"].nunique())).to_string(index=False))
    else:
        print("No errors.")

    df.to_csv(args.output_dir / "predictions_with_language.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "summary_by_language.csv", index=False, encoding="utf-8-sig")
    class_table.to_csv(args.output_dir / "class_f1_by_language.csv", index=False, encoding="utf-8-sig")
    confusions.to_csv(args.output_dir / "top_confusions_by_language.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "overall.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
