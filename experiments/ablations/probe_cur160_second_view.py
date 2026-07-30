from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import train_qwen_distill_v12 as v12
import train_qwen_segment_v4 as v4
from feature_utils_qwen_v4 import ACTION_TO_FAMILY, ALL_CLASSES, FAMILY_NAMES, STRUCTURED_DIM


NUM_CLASSES = len(ALL_CLASSES)
FAMILY_INDEX = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)

VIEW_BUDGETS: Dict[str, Dict[str, int]] = {
    "A_current_heavy": {
        "history": 32,
        "action": 64,
        "meta": 24,
        "current": 136,
    },
    "B_history_heavy": {
        "history": 88,
        "action": 48,
        "meta": 24,
        "current": 96,
    },
    "C_action_heavy": {
        "history": 40,
        "action": 112,
        "meta": 24,
        "current": 80,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate alternate token-budget views on the genuine fold-0 "
            "cur160 student checkpoint. The saved base validation logits are "
            "reused, so only alternate views require GPU inference."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("model/distill_cur160_eval"),
    )
    parser.add_argument("--eval-fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tokenize-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--tree-oof",
        type=Path,
        default=None,
        help=(
            "Optional 70,000-row genuine OOF tree probability array. "
            "It is subset to this validation fold and blended leak-free."
        ),
    )
    parser.add_argument("--tree-weight", type=float, default=0.15)
    parser.add_argument(
        "--margin-grid",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.30],
    )
    parser.add_argument(
        "--blend-grid",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75],
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=list(VIEW_BUDGETS),
        default=list(VIEW_BUDGETS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/cur160_second_view_probe"),
    )
    return parser.parse_args()


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        f1_score(
            labels,
            predictions,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0,
        )
    )


def class_f1(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return f1_score(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
        average=None,
        zero_division=0,
    )


def stable_log_softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    shifted = values - values.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def stable_softmax_from_log(log_probability: np.ndarray) -> np.ndarray:
    shifted = log_probability - log_probability.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def final_logits(
    action_logits: np.ndarray,
    family_logits: np.ndarray,
    postprocess: dict,
) -> np.ndarray:
    class_weights = np.asarray(
        postprocess["training_class_weights"],
        dtype=np.float64,
    )
    return (
        action_logits.astype(np.float64)
        / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"])
        * family_logits.astype(np.float64)[:, FAMILY_INDEX]
        - float(postprocess["prior_beta"])
        * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )


def apply_tree_blend(
    qwen_log_probability: np.ndarray,
    tree_probability: np.ndarray | None,
    tree_weight: float,
) -> np.ndarray:
    if tree_probability is None:
        return qwen_log_probability

    return (
        (1.0 - tree_weight) * qwen_log_probability
        + tree_weight
        * np.log(np.maximum(tree_probability, 1e-12))
    )


def load_base_logits(
    model_dir: Path,
    expected_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    path = model_dir / "validation_logits_v12.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    package = np.load(path)
    action_logits = package["action_logits"].astype(np.float32)
    family_logits = package["family_logits"].astype(np.float32)
    labels = package["labels"].astype(np.int64)

    if not np.array_equal(labels, expected_labels):
        raise RuntimeError(
            "Saved validation logits do not align with reconstructed fold labels."
        )

    return action_logits, family_logits


def make_view_loader(
    samples: List[dict],
    labels: np.ndarray,
    validation_indices: np.ndarray,
    tokenizer,
    budgets: Dict[str, int],
    batch_size: int,
    num_workers: int,
    chunk_size: int,
    description: str,
) -> DataLoader:
    if sum(budgets.values()) != 256:
        raise ValueError(
            f"{description} budgets must sum to 256, got {sum(budgets.values())}"
        )

    original_budgets = dict(v4.SEGMENT_BUDGETS)
    original_max_length = int(v4.MAX_LENGTH)

    try:
        v4.SEGMENT_BUDGETS.clear()
        v4.SEGMENT_BUDGETS.update(budgets)
        v4.MAX_LENGTH = 256

        validation_samples = [
            samples[int(index)]
            for index in validation_indices
        ]
        validation_labels = labels[validation_indices]

        dataset = v4.EncodedSegmentDataset(
            validation_samples,
            validation_labels,
            tokenizer,
            chunk_size=chunk_size,
            description=description,
        )
    finally:
        v4.SEGMENT_BUDGETS.clear()
        v4.SEGMENT_BUDGETS.update(original_budgets)
        v4.MAX_LENGTH = original_max_length

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=v4.SegmentCollator(tokenizer.pad_token_id),
        num_workers=num_workers,
        pin_memory=True,
    )


def evaluate_view(
    model: v12.QwenSegmentClassifier,
    loader: DataLoader,
    device: torch.device,
    postprocess: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_weights = torch.tensor(
        postprocess["training_class_weights"],
        dtype=torch.float32,
        device=device,
    )
    family_index = torch.tensor(
        FAMILY_INDEX,
        dtype=torch.long,
        device=device,
    )

    evaluation = v12.evaluate(
        model,
        loader,
        device,
        postprocess,
        class_weights,
        family_index,
    )
    return (
        evaluation["action_logits"].astype(np.float32),
        evaluation["family_logits"].astype(np.float32),
        evaluation["labels"].astype(np.int64),
    )


def disagreement_stats(
    labels: np.ndarray,
    base_predictions: np.ndarray,
    alternate_predictions: np.ndarray,
) -> dict:
    disagree = base_predictions != alternate_predictions
    count = int(disagree.sum())

    if count == 0:
        return {
            "disagreement_rows": 0,
            "base_correct_on_disagreement": 0,
            "alternate_correct_on_disagreement": 0,
            "alternate_win_rate": None,
        }

    base_correct = int(
        np.sum(base_predictions[disagree] == labels[disagree])
    )
    alternate_correct = int(
        np.sum(alternate_predictions[disagree] == labels[disagree])
    )

    decisive = base_correct + alternate_correct
    win_rate = (
        alternate_correct / decisive
        if decisive > 0
        else None
    )

    return {
        "disagreement_rows": count,
        "base_correct_on_disagreement": base_correct,
        "alternate_correct_on_disagreement": alternate_correct,
        "alternate_win_rate": win_rate,
    }


def screen_view(
    view_name: str,
    labels: np.ndarray,
    base_log_probability: np.ndarray,
    alternate_log_probability: np.ndarray,
    margin_grid: List[float],
    blend_grid: List[float],
) -> Tuple[List[dict], dict]:
    base_probability = stable_softmax_from_log(base_log_probability)
    base_predictions = base_log_probability.argmax(axis=1)
    base_score = macro_f1(labels, base_predictions)

    sorted_probability = np.sort(base_probability, axis=1)
    margin = sorted_probability[:, -1] - sorted_probability[:, -2]

    alternate_predictions = alternate_log_probability.argmax(axis=1)
    direct_stats = disagreement_stats(
        labels,
        base_predictions,
        alternate_predictions,
    )

    rows: List[dict] = []
    for threshold in margin_grid:
        eligible = margin <= threshold

        for blend_weight in blend_grid:
            blended = (
                (1.0 - blend_weight) * base_log_probability
                + blend_weight * alternate_log_probability
            )
            candidate = base_predictions.copy()
            candidate[eligible] = blended[eligible].argmax(axis=1)

            score = macro_f1(labels, candidate)
            per_class_delta = (
                class_f1(labels, candidate)
                - class_f1(labels, base_predictions)
            )

            rows.append(
                {
                    "view": view_name,
                    "margin_threshold": float(threshold),
                    "blend_weight": float(blend_weight),
                    "macro_f1": float(score),
                    "gain": float(score - base_score),
                    "eligible_rows": int(eligible.sum()),
                    "changed_rows": int(
                        np.sum(candidate != base_predictions)
                    ),
                    "positive_classes": int(
                        np.sum(per_class_delta > 0)
                    ),
                    "negative_classes": int(
                        np.sum(per_class_delta < 0)
                    ),
                    "worst_class_delta": float(per_class_delta.min()),
                    "class_deltas": {
                        name: float(per_class_delta[index])
                        for index, name in enumerate(ALL_CLASSES)
                    },
                }
            )

    rows.sort(key=lambda row: row["macro_f1"], reverse=True)
    return rows, direct_stats


def neighborhood_stability(
    all_rows: List[dict],
    best: dict,
) -> dict:
    same_view = [
        row for row in all_rows
        if row["view"] == best["view"]
    ]

    nearby = [
        row for row in same_view
        if abs(row["margin_threshold"] - best["margin_threshold"]) <= 0.05 + 1e-12
        and abs(row["blend_weight"] - best["blend_weight"]) <= 0.25 + 1e-12
    ]

    gains = np.asarray(
        [row["gain"] for row in nearby],
        dtype=np.float64,
    )

    return {
        "nearby_configs": int(len(nearby)),
        "nearby_positive": int(np.sum(gains > 0)),
        "nearby_min_gain": float(gains.min()) if len(gains) else None,
        "nearby_mean_gain": float(gains.mean()) if len(gains) else None,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Load data and reconstruct validation split...")
    samples, labels = v12.load_data(args.data, args.labels)
    _, validation_indices, _ = v12.make_split(
        samples,
        labels,
        args.eval_fold,
    )
    validation_labels = labels[validation_indices]

    metadata = json.loads(
        (args.model_dir / "metadata.json").read_text(encoding="utf-8")
    )
    postprocess = json.loads(
        (args.model_dir / "postprocess.json").read_text(encoding="utf-8")
    )

    base_action_logits, base_family_logits = load_base_logits(
        args.model_dir,
        validation_labels,
    )
    base_qwen_log_probability = stable_log_softmax(
        final_logits(
            base_action_logits,
            base_family_logits,
            postprocess,
        )
    )

    tree_probability = None
    if args.tree_oof is not None:
        full_tree_probability = np.load(args.tree_oof).astype(np.float64)
        if full_tree_probability.shape != (len(samples), NUM_CLASSES):
            raise RuntimeError(
                f"Expected tree OOF shape {(len(samples), NUM_CLASSES)}, "
                f"got {full_tree_probability.shape}"
            )
        tree_probability = full_tree_probability[validation_indices]
        print(
            f"Use genuine tree OOF: rows={len(tree_probability)}, "
            f"weight={args.tree_weight:.3f}"
        )

    base_log_probability = apply_tree_blend(
        base_qwen_log_probability,
        tree_probability,
        args.tree_weight,
    )
    base_predictions = base_log_probability.argmax(axis=1)
    base_score = macro_f1(validation_labels, base_predictions)
    print(f"Base fold-{args.eval_fold} Macro-F1: {base_score:.6f}")

    model_name = str(metadata["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir),
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Load trained cur160 fold checkpoint...")
    model = v12.load_model_from_v4(
        args.model_dir,
        model_name,
        tokenizer,
        device,
        train_lora=False,
    )
    model.eval()

    all_rows: List[dict] = []
    direct_stats_by_view: Dict[str, dict] = {}
    saved_arrays = {
        "labels": validation_labels.astype(np.int64),
        "validation_indices": validation_indices.astype(np.int64),
        "base_action_logits": base_action_logits.astype(np.float32),
        "base_family_logits": base_family_logits.astype(np.float32),
    }

    for view_name in args.views:
        budgets = VIEW_BUDGETS[view_name]
        print()
        print("=" * 72)
        print(view_name, budgets)
        print("=" * 72)

        loader = make_view_loader(
            samples=samples,
            labels=labels,
            validation_indices=validation_indices,
            tokenizer=tokenizer,
            budgets=budgets,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            chunk_size=args.tokenize_chunk_size,
            description=f"Tokenize {view_name}",
        )

        action_logits, family_logits, view_labels = evaluate_view(
            model,
            loader,
            device,
            postprocess,
        )
        if not np.array_equal(view_labels, validation_labels):
            raise RuntimeError(f"{view_name}: label order mismatch")

        saved_arrays[f"{view_name}_action_logits"] = action_logits
        saved_arrays[f"{view_name}_family_logits"] = family_logits

        alternate_qwen_log_probability = stable_log_softmax(
            final_logits(
                action_logits,
                family_logits,
                postprocess,
            )
        )
        alternate_log_probability = apply_tree_blend(
            alternate_qwen_log_probability,
            tree_probability,
            args.tree_weight,
        )

        rows, direct_stats = screen_view(
            view_name,
            validation_labels,
            base_log_probability,
            alternate_log_probability,
            args.margin_grid,
            args.blend_grid,
        )
        direct_stats_by_view[view_name] = direct_stats
        all_rows.extend(rows)

        print("Direct disagreement stats:")
        print(json.dumps(direct_stats, indent=2))
        print("Best configurations:")
        for row in rows[:5]:
            print(
                f"  margin={row['margin_threshold']:.2f} "
                f"blend={row['blend_weight']:.2f} "
                f"gain={row['gain']:+.6f} "
                f"eligible={row['eligible_rows']} "
                f"changed={row['changed_rows']} "
                f"worst_class={row['worst_class_delta']:+.6f}"
            )

    all_rows.sort(key=lambda row: row["macro_f1"], reverse=True)
    best = all_rows[0]
    stability = neighborhood_stability(all_rows, best)

    strong = (
        best["gain"] >= 0.0015
        and stability["nearby_positive"] >= max(
            3,
            int(np.ceil(stability["nearby_configs"] * 0.75)),
        )
        and stability["nearby_min_gain"] is not None
        and stability["nearby_min_gain"] > 0
        and best["changed_rows"] >= 20
    )

    report = {
        "base_macro_f1": base_score,
        "tree_oof_used": args.tree_oof is not None,
        "tree_weight": args.tree_weight if args.tree_oof is not None else 0.0,
        "best": best,
        "stability": stability,
        "direct_stats_by_view": direct_stats_by_view,
        "verdict": "PROMISING" if strong else "DISCARD",
        "top_configs": all_rows[:50],
        "view_budgets": VIEW_BUDGETS,
    }

    np.savez_compressed(
        args.output_dir / "second_view_logits_fold0.npz",
        **saved_arrays,
    )
    (args.output_dir / "second_view_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("FINAL RESULT")
    print("=" * 72)
    print("Best:")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print("Stability:")
    print(json.dumps(stability, ensure_ascii=False, indent=2))
    print("Verdict:", report["verdict"])
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
