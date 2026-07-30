from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.special import softmax
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
SEED = 42

ACTION_TO_FAMILY = np.asarray(
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 3],
    dtype=np.int64,
)

PAIR_SPECS = {
    "read_vs_grep": (LABEL2ID["read_file"], LABEL2ID["grep_search"]),
    "read_vs_list": (LABEL2ID["read_file"], LABEL2ID["list_directory"]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leak-free fold0 probe for targeted file/search binary rerankers. "
            "Train TF-IDF logistic specialists on fold0 train and override only "
            "when Qwen top-2 is the exact target pair."
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
        default=Path("model/v32_file_pair_reranker_probe"),
    )
    parser.add_argument("--eval-fold", type=int, default=0)
    parser.add_argument("--word-features", type=int, default=50000)
    parser.add_argument("--char-features", type=int, default=50000)
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


def load_data(
    data_path: Path,
    labels_path: Path,
) -> Tuple[List[dict], np.ndarray]:
    samples: List[dict] = []
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

    raise ValueError(f"Invalid eval fold: {eval_fold}")


def compact_json(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_text(sample: dict) -> str:
    current = compact_json(sample.get("current_prompt", ""))
    history = compact_json(sample.get("history", []))
    meta = compact_json(sample.get("session_meta", {}))
    workspace = compact_json(sample.get("workspace", {}))

    return (
        "[CURRENT]\n"
        + current
        + "\n[HISTORY]\n"
        + history
        + "\n[META]\n"
        + meta
        + "\n[WORKSPACE]\n"
        + workspace
    )


def build_final_logits(
    action_logits: np.ndarray,
    family_logits: np.ndarray,
    postprocess: dict,
) -> np.ndarray:
    class_weights = np.asarray(
        postprocess["training_class_weights"],
        dtype=np.float64,
    )
    return (
        action_logits.astype(np.float64) / 0.6
        + 0.15 * family_logits.astype(np.float64)[:, ACTION_TO_FAMILY]
        - 0.25 * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )


def fit_pair_model(
    train_texts: List[str],
    train_labels: np.ndarray,
    pair: Tuple[int, int],
    word_features: int,
    char_features: int,
    min_df: int,
):
    pair_mask = np.isin(train_labels, np.asarray(pair))
    pair_texts = [text for text, keep in zip(train_texts, pair_mask) if keep]
    pair_labels = train_labels[pair_mask]

    y_binary = (pair_labels == pair[1]).astype(np.int64)

    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=word_features,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=min_df,
        max_features=char_features,
        sublinear_tf=True,
        dtype=np.float32,
    )

    x_word = word.fit_transform(pair_texts)
    x_char = char.fit_transform(pair_texts)

    from scipy.sparse import hstack

    x_train = hstack([x_word, x_char], format="csr")

    model = LogisticRegression(
        C=2.0,
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
        random_state=SEED,
    )
    model.fit(x_train, y_binary)
    return word, char, model, int(pair_mask.sum())


def predict_pair(
    texts: List[str],
    word,
    char,
    model,
    pair: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.sparse import hstack

    x = hstack(
        [word.transform(texts), char.transform(texts)],
        format="csr",
    )
    prob_second = model.predict_proba(x)[:, 1]
    pred = np.where(prob_second >= 0.5, pair[1], pair[0]).astype(np.int64)
    confidence = np.maximum(prob_second, 1.0 - prob_second)
    return pred, confidence


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples, labels = load_data(args.data, args.labels)
    train_idx, val_idx = make_split(samples, labels, args.eval_fold)

    npz = np.load(args.logits)
    action_logits = npz["action_logits"].astype(np.float32)
    family_logits = npz["family_logits"].astype(np.float32)
    y_val = npz["labels"].astype(np.int64)

    if len(val_idx) != len(y_val):
        raise RuntimeError(
            f"Split/logit mismatch: val_idx={len(val_idx)}, logits={len(y_val)}"
        )
    if not np.array_equal(labels[val_idx], y_val):
        raise RuntimeError("Reconstructed fold labels do not match V12 NPZ labels.")

    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))
    final_logits = build_final_logits(action_logits, family_logits, postprocess)
    qwen_pred = final_logits.argmax(axis=1).astype(np.int64)
    qwen_prob = softmax(final_logits, axis=1)

    top2 = np.argsort(final_logits, axis=1)[:, -2:][:, ::-1]
    qwen_margin = (
        np.take_along_axis(final_logits, top2[:, 0:1], axis=1)[:, 0]
        - np.take_along_axis(final_logits, top2[:, 1:2], axis=1)[:, 0]
    )

    train_texts = [build_text(samples[int(i)]) for i in train_idx]
    val_texts = [build_text(samples[int(i)]) for i in val_idx]
    train_labels = labels[train_idx]

    baseline = macro_f1(y_val, qwen_pred)
    print(f"Qwen baseline Macro-F1: {baseline:.9f}")

    pair_outputs = {}
    pair_reports = {}

    for name, pair in PAIR_SPECS.items():
        print(f"\nTrain specialist: {name} {CLASSES[pair[0]]} vs {CLASSES[pair[1]]}")
        word, char, model, train_count = fit_pair_model(
            train_texts,
            train_labels,
            pair,
            args.word_features,
            args.char_features,
            args.min_df,
        )
        pair_pred, pair_conf = predict_pair(
            val_texts,
            word,
            char,
            model,
            pair,
        )

        exact_pair_top2 = (
            np.isin(top2[:, 0], pair)
            & np.isin(top2[:, 1], pair)
            & (top2[:, 0] != top2[:, 1])
        )

        pair_val_truth = np.isin(y_val, pair)
        specialist_accuracy_on_true_pair = float(
            (pair_pred[pair_val_truth] == y_val[pair_val_truth]).mean()
        )

        pair_outputs[name] = {
            "pair": pair,
            "pred": pair_pred,
            "confidence": pair_conf,
            "gate": exact_pair_top2,
        }
        pair_reports[name] = {
            "pair": [CLASSES[pair[0]], CLASSES[pair[1]]],
            "train_count": train_count,
            "top2_gate_count": int(exact_pair_top2.sum()),
            "specialist_accuracy_on_true_pair": specialist_accuracy_on_true_pair,
        }

        print(f"  train pair samples:       {train_count}")
        print(f"  exact top2 gate samples:  {exact_pair_top2.sum()}")
        print(f"  specialist pair accuracy: {specialist_accuracy_on_true_pair:.6f}")

    margin_thresholds = [
        0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
        0.75, 1.00, 1.50, 2.00, 3.00, 5.00,
    ]
    confidence_thresholds = [
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95,
    ]

    search_rows = []
    best = {
        "macro_f1": baseline,
        "gain": 0.0,
        "margin_threshold": None,
        "confidence_threshold": None,
        "overrides": 0,
        "corrected": 0,
        "damaged": 0,
    }

    for margin_threshold in margin_thresholds:
        for confidence_threshold in confidence_thresholds:
            pred = qwen_pred.copy()
            override_mask_all = np.zeros(len(y_val), dtype=bool)

            for output in pair_outputs.values():
                gate = (
                    output["gate"]
                    & (qwen_margin <= margin_threshold)
                    & (output["confidence"] >= confidence_threshold)
                )
                pred[gate] = output["pred"][gate]
                override_mask_all |= gate

            score = macro_f1(y_val, pred)
            corrected = int(
                ((qwen_pred != y_val) & (pred == y_val) & override_mask_all).sum()
            )
            damaged = int(
                ((qwen_pred == y_val) & (pred != y_val) & override_mask_all).sum()
            )
            overrides = int((pred != qwen_pred).sum())

            row = {
                "macro_f1": score,
                "gain": score - baseline,
                "margin_threshold": margin_threshold,
                "confidence_threshold": confidence_threshold,
                "overrides": overrides,
                "corrected": corrected,
                "damaged": damaged,
                "net_sample_gain": corrected - damaged,
            }
            search_rows.append(row)

            if score > best["macro_f1"]:
                best = row

    search_rows.sort(key=lambda row: row["macro_f1"], reverse=True)

    # Also evaluate each pair independently using the jointly best thresholds.
    independent = {}
    if best["margin_threshold"] is not None:
        for name, output in pair_outputs.items():
            pred = qwen_pred.copy()
            gate = (
                output["gate"]
                & (qwen_margin <= best["margin_threshold"])
                & (output["confidence"] >= best["confidence_threshold"])
            )
            pred[gate] = output["pred"][gate]
            independent[name] = {
                "macro_f1": macro_f1(y_val, pred),
                "gain": macro_f1(y_val, pred) - baseline,
                "overrides": int((pred != qwen_pred).sum()),
                "corrected": int(
                    ((qwen_pred != y_val) & (pred == y_val) & gate).sum()
                ),
                "damaged": int(
                    ((qwen_pred == y_val) & (pred != y_val) & gate).sum()
                ),
            }

    result = {
        "baseline_macro_f1": baseline,
        "pair_reports": pair_reports,
        "best": best,
        "independent_at_best_thresholds": independent,
        "top_results": search_rows[:30],
        "note": (
            "This is a leak-free fold0 probe. Specialists train only on fold0 train. "
            "Overrides occur only when Qwen top-2 is exactly the target pair."
        ),
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
        writer = csv.DictWriter(file, fieldnames=list(search_rows[0].keys()))
        writer.writeheader()
        writer.writerows(search_rows)

    print("\n" + "=" * 80)
    print("PAIR RERANKER RESULT")
    print("=" * 80)
    print(f"Baseline: {baseline:.9f}")
    print(
        f"Best:     {best['macro_f1']:.9f} "
        f"({best['gain']:+.9f})"
    )
    print(
        f"Gate: margin<={best['margin_threshold']} "
        f"specialist_conf>={best['confidence_threshold']}"
    )
    print(
        f"Overrides={best['overrides']} "
        f"corrected={best['corrected']} "
        f"damaged={best['damaged']} "
        f"net={best['net_sample_gain']}"
    )

    print("\nIndependent pair gains at best thresholds:")
    for name, row in independent.items():
        print(
            f"  {name:16s} {row['macro_f1']:.9f} "
            f"({row['gain']:+.9f}) "
            f"overrides={row['overrides']} "
            f"corrected={row['corrected']} damaged={row['damaged']}"
        )

    print("\nSaved:", args.output_dir)


if __name__ == "__main__":
    main()
