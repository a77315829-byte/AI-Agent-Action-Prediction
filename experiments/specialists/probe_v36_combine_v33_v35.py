from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
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
ASK_ID = LABEL2ID["ask_user"]
PLAN_ID = LABEL2ID["plan_task"]

SEED = 42

ACTION_TO_FAMILY = np.asarray(
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 3],
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine fixed V33 read/list and V35 ask/plan specialists on Qwen and V23."
    )
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument(
        "--qwen-logits",
        type=Path,
        default=Path("model/qwen_distill_v12_eval/validation_logits_v12.npz"),
    )
    parser.add_argument(
        "--postprocess",
        type=Path,
        default=Path("model/qwen_distill_v12_eval/postprocess.json"),
    )
    parser.add_argument(
        "--v23-dirs",
        type=Path,
        nargs="*",
        default=[
            Path("model/v23_tree_tfidf_structured_probe"),
            Path("model/v23_tree_tfidf_structured_probe_light"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/v36_v23_plus_v33_v35_probe"),
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

    raise ValueError(f"Invalid eval fold: {eval_fold}")


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


def build_qwen_logits(
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


def load_prediction(path: Path, expected_size: int) -> np.ndarray:
    arr = np.asarray(np.load(path))
    if arr.ndim == 2:
        arr = arr.argmax(axis=1)
    arr = arr.reshape(-1).astype(np.int64)
    if len(arr) != expected_size:
        raise ValueError(f"Prediction length mismatch: {path}")
    return arr


def fit_pair_model(
    train_texts_all: List[str],
    y_train: np.ndarray,
    pair: Tuple[int, int],
    c_value: float,
    word_features: int,
    char_features: int,
    min_df: int,
):
    mask = np.isin(y_train, pair)
    texts = [text for text, keep in zip(train_texts_all, mask) if keep]
    y = (y_train[mask] == pair[1]).astype(np.int64)

    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=min_df,
        max_features=word_features,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        min_df=min_df,
        max_features=char_features,
        sublinear_tf=True,
        dtype=np.float32,
    )

    x_train = hstack(
        [word.fit_transform(texts), char.fit_transform(texts)],
        format="csr",
    )

    model = LogisticRegression(
        C=c_value,
        max_iter=1500,
        solver="liblinear",
        class_weight="balanced",
        random_state=SEED,
    )
    model.fit(x_train, y)
    return word, char, model


def predict_pair(
    val_texts: List[str],
    word,
    char,
    model,
) -> np.ndarray:
    x_val = hstack(
        [word.transform(val_texts), char.transform(val_texts)],
        format="csr",
    )
    return model.predict_proba(x_val)[:, 1]


def evaluate(
    baseline_pred: np.ndarray,
    target_pred: np.ndarray,
    gate: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    combined = baseline_pred.copy()
    combined[gate] = target_pred[gate]
    changed = combined != baseline_pred

    baseline_score = macro_f1(y_true, baseline_pred)
    score = macro_f1(y_true, combined)

    corrected = int(
        ((baseline_pred != y_true) & (combined == y_true) & changed).sum()
    )
    damaged = int(
        ((baseline_pred == y_true) & (combined != y_true) & changed).sum()
    )

    return {
        "baseline_macro_f1": baseline_score,
        "combined_macro_f1": score,
        "gain": score - baseline_score,
        "overrides": int(changed.sum()),
        "corrected": corrected,
        "damaged": damaged,
        "net": corrected - damaged,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples, labels = load_data(args.data, args.labels)
    train_idx, val_idx = make_split(samples, labels, args.eval_fold)

    npz = np.load(args.qwen_logits)
    action_logits = npz["action_logits"].astype(np.float32)
    family_logits = npz["family_logits"].astype(np.float32)
    y_val = npz["labels"].astype(np.int64)

    if not np.array_equal(labels[val_idx], y_val):
        raise RuntimeError("Validation split does not match Qwen NPZ.")

    pp = json.loads(args.postprocess.read_text(encoding="utf-8"))
    qwen_logits = build_qwen_logits(action_logits, family_logits, pp)
    qwen_pred = qwen_logits.argmax(axis=1).astype(np.int64)

    top2 = np.argsort(qwen_logits, axis=1)[:, -2:][:, ::-1]
    qwen_margin = (
        np.take_along_axis(qwen_logits, top2[:, 0:1], axis=1)[:, 0]
        - np.take_along_axis(qwen_logits, top2[:, 1:2], axis=1)[:, 0]
    )

    train_texts = [build_text(samples[int(i)]) for i in train_idx]
    val_texts = [build_text(samples[int(i)]) for i in val_idx]
    y_train = labels[train_idx]

    # V33 fixed configuration
    rl_word, rl_char, rl_model = fit_pair_model(
        train_texts,
        y_train,
        (READ_ID, LIST_ID),
        c_value=1.0,
        word_features=args.word_features,
        char_features=args.char_features,
        min_df=args.min_df,
    )
    prob_list = predict_pair(val_texts, rl_word, rl_char, rl_model)
    prob_read = 1.0 - prob_list

    rl_exact_pair = (
        np.isin(top2[:, 0], [READ_ID, LIST_ID])
        & np.isin(top2[:, 1], [READ_ID, LIST_ID])
    )
    list_to_read = (
        rl_exact_pair
        & (qwen_pred == LIST_ID)
        & (qwen_margin <= 0.75)
        & (prob_read >= 0.50)
    )
    read_to_list = (
        rl_exact_pair
        & (qwen_pred == READ_ID)
        & (qwen_margin <= 0.75)
        & (prob_list >= 0.50)
    )
    v33_gate = list_to_read | read_to_list
    v33_target = qwen_pred.copy()
    v33_target[list_to_read] = READ_ID
    v33_target[read_to_list] = LIST_ID

    # V35 fixed configuration
    ap_word, ap_char, ap_model = fit_pair_model(
        train_texts,
        y_train,
        (ASK_ID, PLAN_ID),
        c_value=0.25,
        word_features=args.word_features,
        char_features=args.char_features,
        min_df=args.min_df,
    )
    prob_plan = predict_pair(val_texts, ap_word, ap_char, ap_model)

    ap_exact_pair = (
        np.isin(top2[:, 0], [ASK_ID, PLAN_ID])
        & np.isin(top2[:, 1], [ASK_ID, PLAN_ID])
    )
    ask_to_plan = (
        ap_exact_pair
        & (qwen_pred == ASK_ID)
        & (qwen_margin <= 1.5)
        & (prob_plan >= 0.55)
    )
    v35_gate = ask_to_plan
    v35_target = qwen_pred.copy()
    v35_target[ask_to_plan] = PLAN_ID

    # No overlap is expected because the class pairs are disjoint.
    overlap = v33_gate & v35_gate
    if overlap.any():
        raise RuntimeError(f"Unexpected specialist gate overlap: {overlap.sum()}")

    combined_gate = v33_gate | v35_gate
    combined_target = qwen_pred.copy()
    combined_target[v33_gate] = v33_target[v33_gate]
    combined_target[v35_gate] = v35_target[v35_gate]

    baselines = {"qwen": qwen_pred}
    for v23_dir in args.v23_dirs:
        pred_path = v23_dir / "pred_blend_best.npy"
        idx_path = v23_dir / "val_indices.npy"

        if not pred_path.exists():
            continue

        if idx_path.exists():
            saved_idx = np.load(idx_path).reshape(-1).astype(np.int64)
            if not np.array_equal(saved_idx, val_idx):
                raise RuntimeError(f"Validation index mismatch: {v23_dir}")

        baselines[v23_dir.name] = load_prediction(pred_path, len(y_val))

    results = {}

    print("=" * 88)
    print("V36: V33 + V35 COMBINED SPECIALISTS")
    print("=" * 88)
    print(f"V33 gate samples: {v33_gate.sum()}")
    print(f"V35 gate samples: {v35_gate.sum()}")
    print(f"Combined gates:   {combined_gate.sum()}")

    for name, baseline_pred in baselines.items():
        v33_result = evaluate(
            baseline_pred,
            v33_target,
            v33_gate,
            y_val,
        )
        v35_result = evaluate(
            baseline_pred,
            v35_target,
            v35_gate,
            y_val,
        )
        combined_result = evaluate(
            baseline_pred,
            combined_target,
            combined_gate,
            y_val,
        )

        results[name] = {
            "v33_only": v33_result,
            "v35_only": v35_result,
            "combined": combined_result,
        }

        print()
        print(f"[{name}]")
        print(
            f"  Baseline: {combined_result['baseline_macro_f1']:.9f}"
        )
        print(
            f"  V33 only: {v33_result['combined_macro_f1']:.9f} "
            f"({v33_result['gain']:+.9f})"
        )
        print(
            f"  V35 only: {v35_result['combined_macro_f1']:.9f} "
            f"({v35_result['gain']:+.9f})"
        )
        print(
            f"  Combined: {combined_result['combined_macro_f1']:.9f} "
            f"({combined_result['gain']:+.9f})"
        )
        print(
            f"  Overrides={combined_result['overrides']} "
            f"corrected={combined_result['corrected']} "
            f"damaged={combined_result['damaged']} "
            f"net={combined_result['net']}"
        )

    report = {
        "fixed_configs": {
            "v33": {
                "pair": ["read_file", "list_directory"],
                "C": 1.0,
                "direction": "both",
                "max_margin": 0.75,
                "min_probability": 0.50,
            },
            "v35": {
                "pair": ["ask_user", "plan_task"],
                "C": 0.25,
                "direction": "ask_to_plan_only",
                "max_margin": 1.5,
                "min_probability": 0.55,
            },
        },
        "gate_counts": {
            "v33": int(v33_gate.sum()),
            "v35": int(v35_gate.sum()),
            "combined": int(combined_gate.sum()),
        },
        "results": results,
    }

    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
