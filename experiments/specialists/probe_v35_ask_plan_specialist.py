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
ASK_ID = LABEL2ID["ask_user"]
PLAN_ID = LABEL2ID["plan_task"]
SEED = 42

ACTION_TO_FAMILY = np.asarray(
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 3],
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leak-free ask_user vs plan_task specialist probe. "
            "Evaluates on Qwen and optional V23 blend predictions."
        )
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
        default=Path("model/v35_ask_plan_specialist_probe"),
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


def compact(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_text(sample: dict) -> str:
    current = compact(sample.get("current_prompt", ""))
    history = compact(sample.get("history", []))
    meta = compact(sample.get("session_meta", {}))
    workspace = compact(sample.get("workspace", {}))

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
        if arr.shape[1] != len(CLASSES):
            raise ValueError(f"Unexpected prediction shape: {path} {arr.shape}")
        arr = arr.argmax(axis=1)
    arr = arr.reshape(-1).astype(np.int64)
    if len(arr) != expected_size:
        raise ValueError(f"Prediction length mismatch: {path}")
    return arr


def evaluate_candidate(
    baseline_pred: np.ndarray,
    y_true: np.ndarray,
    specialist_target: np.ndarray,
    gate: np.ndarray,
) -> dict:
    combined = baseline_pred.copy()
    combined[gate] = specialist_target[gate]

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
        raise RuntimeError("Reconstructed validation labels do not match Qwen NPZ.")

    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))
    qwen_logits = build_qwen_logits(action_logits, family_logits, postprocess)
    qwen_pred = qwen_logits.argmax(axis=1).astype(np.int64)

    top2 = np.argsort(qwen_logits, axis=1)[:, -2:][:, ::-1]
    qwen_margin = (
        np.take_along_axis(qwen_logits, top2[:, 0:1], axis=1)[:, 0]
        - np.take_along_axis(qwen_logits, top2[:, 1:2], axis=1)[:, 0]
    )

    exact_pair = (
        np.isin(top2[:, 0], [ASK_ID, PLAN_ID])
        & np.isin(top2[:, 1], [ASK_ID, PLAN_ID])
    )

    train_texts_all = [build_text(samples[int(i)]) for i in train_idx]
    val_texts = [build_text(samples[int(i)]) for i in val_idx]
    y_train = labels[train_idx]

    pair_mask = np.isin(y_train, [ASK_ID, PLAN_ID])
    pair_texts = [
        text for text, keep in zip(train_texts_all, pair_mask)
        if keep
    ]
    pair_y = (y_train[pair_mask] == PLAN_ID).astype(np.int64)

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
            word.fit_transform(pair_texts),
            char.fit_transform(pair_texts),
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

    baselines = {"qwen": qwen_pred}

    for v23_dir in args.v23_dirs:
        pred_path = v23_dir / "pred_blend_best.npy"
        index_path = v23_dir / "val_indices.npy"

        if not pred_path.exists():
            continue

        if index_path.exists():
            saved_idx = np.load(index_path).reshape(-1).astype(np.int64)
            if not np.array_equal(saved_idx, val_idx):
                raise RuntimeError(f"Validation index mismatch: {v23_dir}")

        baselines[v23_dir.name] = load_prediction(
            pred_path,
            expected_size=len(y_val),
        )

    c_values = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    margin_thresholds = [
        0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
        0.75, 1.00, 1.50, 2.00, 3.00, 5.00,
    ]
    probability_thresholds = [
        0.50, 0.55, 0.60, 0.65, 0.70,
        0.75, 0.80, 0.85, 0.90, 0.95,
    ]
    directions = [
        "both",
        "ask_to_plan_only",
        "plan_to_ask_only",
    ]

    rows = []
    best_by_baseline = {
        name: {
            "baseline": name,
            "combined_macro_f1": macro_f1(y_val, pred),
            "gain": 0.0,
        }
        for name, pred in baselines.items()
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

        prob_plan = model.predict_proba(x_val)[:, 1]
        prob_ask = 1.0 - prob_plan

        for direction in directions:
            for max_margin in margin_thresholds:
                for min_probability in probability_thresholds:
                    ask_to_plan = (
                        exact_pair
                        & (qwen_pred == ASK_ID)
                        & (qwen_margin <= max_margin)
                        & (prob_plan >= min_probability)
                    )
                    plan_to_ask = (
                        exact_pair
                        & (qwen_pred == PLAN_ID)
                        & (qwen_margin <= max_margin)
                        & (prob_ask >= min_probability)
                    )

                    if direction == "ask_to_plan_only":
                        gate = ask_to_plan
                    elif direction == "plan_to_ask_only":
                        gate = plan_to_ask
                    else:
                        gate = ask_to_plan | plan_to_ask

                    specialist_target = qwen_pred.copy()
                    specialist_target[ask_to_plan] = PLAN_ID
                    specialist_target[plan_to_ask] = ASK_ID

                    for baseline_name, baseline_pred in baselines.items():
                        result = evaluate_candidate(
                            baseline_pred,
                            y_val,
                            specialist_target,
                            gate,
                        )
                        row = {
                            "baseline": baseline_name,
                            "C": c_value,
                            "direction": direction,
                            "max_margin": max_margin,
                            "min_probability": min_probability,
                            **result,
                        }
                        rows.append(row)

                        if (
                            result["combined_macro_f1"]
                            > best_by_baseline[baseline_name]["combined_macro_f1"]
                        ):
                            best_by_baseline[baseline_name] = row

    rows.sort(
        key=lambda row: (
            row["baseline"],
            -row["combined_macro_f1"],
        )
    )

    summary = {
        "pair_train_samples": int(pair_mask.sum()),
        "exact_qwen_top2_pair_samples": int(exact_pair.sum()),
        "baseline_scores": {
            name: macro_f1(y_val, pred)
            for name, pred in baselines.items()
        },
        "best_by_baseline": best_by_baseline,
    }

    (args.output_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
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

    print("=" * 88)
    print("V35 ASK_USER / PLAN_TASK SPECIALIST")
    print("=" * 88)
    print(f"Pair train samples:       {pair_mask.sum()}")
    print(f"Exact Qwen top2 samples:  {exact_pair.sum()}")

    for baseline_name, best in best_by_baseline.items():
        baseline_score = summary["baseline_scores"][baseline_name]
        print()
        print(f"[{baseline_name}]")
        print(f"  Baseline: {baseline_score:.9f}")
        print(
            f"  Best:     {best['combined_macro_f1']:.9f} "
            f"({best['gain']:+.9f})"
        )

        if best["gain"] > 0:
            print(
                f"  C={best['C']} direction={best['direction']} "
                f"margin<={best['max_margin']} "
                f"prob>={best['min_probability']}"
            )
            print(
                f"  Overrides={best['overrides']} "
                f"corrected={best['corrected']} "
                f"damaged={best['damaged']} "
                f"net={best['net']}"
            )
        else:
            print("  No positive candidate found.")

    print()
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
