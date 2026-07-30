from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from scipy.special import softmax
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold


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
LABEL2ID = {label: i for i, label in enumerate(CLASSES)}
READ_ID = LABEL2ID["read_file"]
LIST_ID = LABEL2ID["list_directory"]
SEED = 42

ACTION_TO_FAMILY = np.asarray(
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 3],
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Directional read_file/list_directory reranker probe. "
            "Trains only on fold0 train and tests asymmetric override gates."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument(
        "--logits",
        type=Path,
        default=Path("model/qwen_distill_v12_eval/validation_logits_v12.npz"),
    )
    parser.add_argument(
        "--postprocess",
        type=Path,
        default=Path("model/qwen_distill_v12_eval/postprocess.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/v33_read_list_directional_probe"),
    )
    parser.add_argument("--eval-fold", type=int, default=0)
    parser.add_argument("--word-features", type=int, default=70000)
    parser.add_argument("--char-features", type=int, default=70000)
    parser.add_argument("--min-df", type=int, default=2)
    return parser.parse_args()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=np.arange(len(CLASSES)),
            average="macro",
            zero_division=0,
        )
    )


def load_data(data_path: Path, labels_path: Path) -> Tuple[List[dict], np.ndarray]:
    samples = []
    with data_path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                samples.append(json.loads(line))

    with labels_path.open(encoding="utf-8", newline="") as file:
        label_map = {
            str(row["id"]): LABEL2ID[str(row["action"])]
            for row in csv.DictReader(file)
        }

    labels = np.asarray(
        [label_map[str(sample["id"])] for sample in samples],
        dtype=np.int64,
    )
    return samples, labels


def make_split(
    samples: Sequence[dict],
    labels: np.ndarray,
    eval_fold: int,
) -> Tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(
        [str(sample["id"]).rsplit("-step_", 1)[0] for sample in samples]
    )
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups)
    ):
        if fold == eval_fold:
            return train_idx.astype(np.int64), val_idx.astype(np.int64)

    raise ValueError(f"Invalid fold: {eval_fold}")


def compact(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_text(sample: dict) -> str:
    return (
        "[CURRENT]\n"
        + compact(sample.get("current_prompt", ""))
        + "\n[HISTORY]\n"
        + compact(sample.get("history", []))
        + "\n[META]\n"
        + compact(sample.get("session_meta", {}))
        + "\n[WORKSPACE]\n"
        + compact(sample.get("workspace", {}))
    )


def final_logits(
    action_logits: np.ndarray,
    family_logits: np.ndarray,
    postprocess: dict,
) -> np.ndarray:
    weights = np.asarray(
        postprocess["training_class_weights"],
        dtype=np.float64,
    )
    return (
        action_logits.astype(np.float64) / 0.6
        + 0.15 * family_logits.astype(np.float64)[:, ACTION_TO_FAMILY]
        - 0.25 * np.log(np.maximum(weights, 1e-12))[None, :]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples, labels = load_data(args.data, args.labels)
    train_idx, val_idx = make_split(samples, labels, args.eval_fold)

    npz = np.load(args.logits)
    action_logits = npz["action_logits"].astype(np.float32)
    family_logits = npz["family_logits"].astype(np.float32)
    y_val = npz["labels"].astype(np.int64)

    if not np.array_equal(labels[val_idx], y_val):
        raise RuntimeError("Fold reconstruction does not match validation logits.")

    pp = json.loads(args.postprocess.read_text(encoding="utf-8"))
    logits = final_logits(action_logits, family_logits, pp)
    qwen_pred = logits.argmax(axis=1).astype(np.int64)
    top2 = np.argsort(logits, axis=1)[:, -2:][:, ::-1]
    margin = (
        np.take_along_axis(logits, top2[:, 0:1], axis=1)[:, 0]
        - np.take_along_axis(logits, top2[:, 1:2], axis=1)[:, 0]
    )

    train_texts_all = [build_text(samples[int(i)]) for i in train_idx]
    val_texts = [build_text(samples[int(i)]) for i in val_idx]
    y_train = labels[train_idx]

    pair_mask = np.isin(y_train, [READ_ID, LIST_ID])
    train_texts = [
        text for text, keep in zip(train_texts_all, pair_mask)
        if keep
    ]
    pair_y = (y_train[pair_mask] == LIST_ID).astype(np.int64)

    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=args.min_df,
        max_features=args.word_features,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        min_df=args.min_df,
        max_features=args.char_features,
        sublinear_tf=True,
        dtype=np.float32,
    )

    x_train = hstack(
        [
            word.fit_transform(train_texts),
            char.fit_transform(train_texts),
        ],
        format="csr",
    )
    x_val = hstack(
        [
            word.transform(val_texts),
            char.transform(val_texts),
        ],
        format="csr",
    )

    baseline = macro_f1(y_val, qwen_pred)
    exact_pair = (
        np.isin(top2[:, 0], [READ_ID, LIST_ID])
        & np.isin(top2[:, 1], [READ_ID, LIST_ID])
    )

    c_values = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    margin_thresholds = [
        0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
        0.75, 1.00, 1.50, 2.00, 3.00, 5.00,
    ]
    probability_thresholds = [
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95,
    ]

    rows = []
    best = {
        "macro_f1": baseline,
        "gain": 0.0,
    }

    for c_value in c_values:
        model = LogisticRegression(
            C=c_value,
            max_iter=1500,
            solver="liblinear",
            class_weight="balanced",
            random_state=SEED,
        )
        model.fit(x_train, pair_y)
        prob_list = model.predict_proba(x_val)[:, 1]
        prob_read = 1.0 - prob_list

        for direction in [
            "list_to_read_only",
            "read_to_list_only",
            "both",
        ]:
            for max_margin in margin_thresholds:
                for min_probability in probability_thresholds:
                    pred = qwen_pred.copy()

                    list_to_read = (
                        exact_pair
                        & (qwen_pred == LIST_ID)
                        & (margin <= max_margin)
                        & (prob_read >= min_probability)
                    )
                    read_to_list = (
                        exact_pair
                        & (qwen_pred == READ_ID)
                        & (margin <= max_margin)
                        & (prob_list >= min_probability)
                    )

                    if direction == "list_to_read_only":
                        gate = list_to_read
                        pred[gate] = READ_ID
                    elif direction == "read_to_list_only":
                        gate = read_to_list
                        pred[gate] = LIST_ID
                    else:
                        gate = list_to_read | read_to_list
                        pred[list_to_read] = READ_ID
                        pred[read_to_list] = LIST_ID

                    score = macro_f1(y_val, pred)
                    changed = pred != qwen_pred
                    corrected = int(
                        ((qwen_pred != y_val) & (pred == y_val) & changed).sum()
                    )
                    damaged = int(
                        ((qwen_pred == y_val) & (pred != y_val) & changed).sum()
                    )

                    row = {
                        "macro_f1": score,
                        "gain": score - baseline,
                        "C": c_value,
                        "direction": direction,
                        "max_margin": max_margin,
                        "min_probability": min_probability,
                        "overrides": int(changed.sum()),
                        "corrected": corrected,
                        "damaged": damaged,
                        "net": corrected - damaged,
                    }
                    rows.append(row)

                    if score > best["macro_f1"]:
                        best = row

    rows.sort(key=lambda row: row["macro_f1"], reverse=True)

    result = {
        "baseline_macro_f1": baseline,
        "pair_train_samples": int(pair_mask.sum()),
        "exact_pair_gate_samples": int(exact_pair.sum()),
        "best": best,
        "top_results": rows[:50],
    }

    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (args.output_dir / "search_results.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("V33 READ/LIST DIRECTIONAL PROBE")
    print("=" * 80)
    print(f"Baseline: {baseline:.9f}")
    print(f"Best:     {best['macro_f1']:.9f} ({best['gain']:+.9f})")
    print(
        f"C={best['C']} direction={best['direction']} "
        f"margin<={best['max_margin']} "
        f"prob>={best['min_probability']}"
    )
    print(
        f"Overrides={best['overrides']} corrected={best['corrected']} "
        f"damaged={best['damaged']} net={best['net']}"
    )
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
