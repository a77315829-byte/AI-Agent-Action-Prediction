import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from feature_utils_qwen_v4 import (
    ACTION_TO_FAMILY,
    ALL_CLASSES,
    build_structured_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--tfidf-predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("model/qwen_tfidf_selector_v1"))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--max-features", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=345)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    samples: List[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True).clip(min=1e-12)


def build_training_weights(
    samples: List[dict],
    labels_path: Path,
    validation_indices: np.ndarray,
) -> np.ndarray:
    frame = pd.read_csv(labels_path)
    label_map = dict(zip(frame["id"].astype(str), frame["action"].astype(str)))
    all_labels = np.asarray(
        [ALL_CLASSES.index(label_map[str(sample["id"])]) for sample in samples],
        dtype=np.int64,
    )
    train_mask = np.ones(len(samples), dtype=bool)
    train_mask[validation_indices] = False
    train_labels = all_labels[train_mask]
    counts = np.bincount(train_labels, minlength=len(ALL_CLASSES)).astype(np.float64)
    return np.power(
        len(train_labels) / (len(ALL_CLASSES) * np.maximum(counts, 1.0)),
        0.25,
    )


def build_meta_features(
    samples: List[dict],
    validation_indices: np.ndarray,
    action_logits: np.ndarray,
    family_logits: np.ndarray,
    tfidf_predictions: np.ndarray,
    global_predictions: np.ndarray,
    training_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    family_index = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)
    final_logits = (
        action_logits / 1.4
        + 1.25 * family_logits[:, family_index]
        + 0.85 * np.log(np.clip(training_weights, 1e-12, None))[None, :]
    )
    qwen_probs = softmax(final_logits)
    raw_probs = softmax(action_logits)
    family_probs = softmax(family_logits)
    qwen_predictions = qwen_probs.argmax(axis=1)

    validation_samples = [samples[int(index)] for index in validation_indices]
    structured = np.stack(
        [build_structured_features(sample) for sample in validation_samples]
    ).astype(np.float32)

    sorted_probs = np.sort(qwen_probs, axis=1)
    confidence_features = np.column_stack(
        [
            qwen_probs.max(axis=1),
            sorted_probs[:, -1] - sorted_probs[:, -2],
            -(qwen_probs * np.log(np.clip(qwen_probs, 1e-12, None))).sum(axis=1),
            (qwen_predictions == tfidf_predictions).astype(np.float32),
            (qwen_predictions == global_predictions).astype(np.float32),
        ]
    ).astype(np.float32)

    action_centered = action_logits - action_logits.mean(axis=1, keepdims=True)
    family_centered = family_logits - family_logits.mean(axis=1, keepdims=True)
    identity = np.eye(len(ALL_CLASSES), dtype=np.float32)

    features = np.hstack(
        [
            qwen_probs,
            raw_probs,
            family_probs,
            action_centered,
            family_centered,
            identity[tfidf_predictions],
            identity[global_predictions],
            structured,
            confidence_features,
        ]
    ).astype(np.float32)
    return features, qwen_predictions, final_logits


def make_model(args: argparse.Namespace, random_state: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(len(ALL_CLASSES))),
            average="macro",
            zero_division=0,
        )
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logits = np.load(args.logits)
    action_logits = logits["action_logits"]
    family_logits = logits["family_logits"]
    labels = logits["labels"].astype(np.int64)
    validation_indices = logits["validation_indices"].astype(np.int64)

    samples = load_jsonl(args.data)
    training_weights = build_training_weights(samples, args.labels, validation_indices)

    predictions_frame = pd.read_csv(args.tfidf_predictions)
    if not np.array_equal(
        predictions_frame["index"].to_numpy(dtype=np.int64), validation_indices
    ):
        raise RuntimeError("TF-IDF validation index order does not match Qwen logits.")

    true_ids = predictions_frame["true_action"].map(ALL_CLASSES.index).to_numpy()
    if not np.array_equal(true_ids, labels):
        raise RuntimeError("TF-IDF labels do not match Qwen labels.")

    tfidf_predictions = predictions_frame["pred_action"].map(ALL_CLASSES.index).to_numpy()
    global_predictions = predictions_frame["global_pred"].map(ALL_CLASSES.index).to_numpy()

    features, qwen_predictions, _ = build_meta_features(
        samples,
        validation_indices,
        action_logits,
        family_logits,
        tfidf_predictions,
        global_predictions,
        training_weights,
    )

    groups = np.asarray(
        [str(samples[int(index)]["id"]).rsplit("-step_", 1)[0] for index in validation_indices]
    )
    disagreement_indices = np.flatnonzero(qwen_predictions != tfidf_predictions)
    disagreement_features = features[disagreement_indices]
    disagreement_groups = groups[disagreement_indices]

    # Positive means TF-IDF is correct while Qwen is wrong.
    selector_targets = (
        (tfidf_predictions[disagreement_indices] == labels[disagreement_indices])
        & (qwen_predictions[disagreement_indices] != labels[disagreement_indices])
    ).astype(np.int64)

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=args.seed,
    )
    selector_probabilities = np.zeros(len(disagreement_indices), dtype=np.float64)
    fold_validation_positions: List[np.ndarray] = []

    for fold, (train_pos, validation_pos) in enumerate(
        splitter.split(disagreement_features, selector_targets, disagreement_groups)
    ):
        model = make_model(args, args.seed + fold)
        model.fit(disagreement_features[train_pos], selector_targets[train_pos])
        selector_probabilities[validation_pos] = model.predict_proba(
            disagreement_features[validation_pos]
        )[:, 1]
        fold_validation_positions.append(validation_pos)

    base_score = macro_f1(labels, qwen_predictions)
    best_score = base_score
    best_threshold = 1.0
    best_count = 0

    for threshold in np.linspace(0.50, 0.90, 81):
        predictions = qwen_predictions.copy()
        selected = disagreement_indices[selector_probabilities >= threshold]
        predictions[selected] = tfidf_predictions[selected]
        score = macro_f1(labels, predictions)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_count = int(len(selected))

    # More conservative estimate: tune threshold without each held fold's groups.
    nested_predictions = qwen_predictions.copy()
    nested_thresholds: List[float] = []
    for held_positions in fold_validation_positions:
        held_group_set = set(disagreement_groups[held_positions].tolist())
        training_sample_mask = np.asarray(
            [group not in held_group_set for group in groups], dtype=bool
        )
        candidate_positions = np.setdiff1d(
            np.arange(len(disagreement_indices)), held_positions, assume_unique=False
        )
        local_best_score = -1.0
        local_best_threshold = 1.0
        for threshold in np.linspace(0.50, 0.90, 81):
            candidate_predictions = qwen_predictions.copy()
            selected = disagreement_indices[
                candidate_positions[
                    selector_probabilities[candidate_positions] >= threshold
                ]
            ]
            candidate_predictions[selected] = tfidf_predictions[selected]
            score = macro_f1(
                labels[training_sample_mask], candidate_predictions[training_sample_mask]
            )
            if score > local_best_score:
                local_best_score = score
                local_best_threshold = float(threshold)

        held_selected = held_positions[
            selector_probabilities[held_positions] >= local_best_threshold
        ]
        held_indices = disagreement_indices[held_selected]
        nested_predictions[held_indices] = tfidf_predictions[held_indices]
        nested_thresholds.append(local_best_threshold)

    nested_score = macro_f1(labels, nested_predictions)

    final_model = make_model(args, args.seed)
    final_model.fit(disagreement_features, selector_targets)
    joblib.dump(
        {
            "model": final_model,
            "threshold": best_threshold,
            "classes": ALL_CLASSES,
            "feature_dim": int(features.shape[1]),
            "postprocess": {
                "family_weight": 1.25,
                "prior_beta": -0.85,
                "action_temperature": 1.4,
            },
        },
        args.output_dir / "selector.joblib",
        compress=3,
    )

    oof_frame = predictions_frame.copy()
    oof_frame["qwen_pred"] = [ALL_CLASSES[index] for index in qwen_predictions]
    oof_frame["selector_probability"] = 0.0
    oof_frame.loc[disagreement_indices, "selector_probability"] = selector_probabilities
    pooled_predictions = qwen_predictions.copy()
    pooled_selected = disagreement_indices[selector_probabilities >= best_threshold]
    pooled_predictions[pooled_selected] = tfidf_predictions[pooled_selected]
    oof_frame["selector_pred"] = [ALL_CLASSES[index] for index in pooled_predictions]
    oof_frame.to_csv(
        args.output_dir / "validation_selector_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report: Dict[str, object] = {
        "qwen_macro_f1": base_score,
        "tfidf_macro_f1": macro_f1(labels, tfidf_predictions),
        "disagreement_samples": int(len(disagreement_indices)),
        "selector_positive_samples": int(selector_targets.sum()),
        "selector_auc": float(roc_auc_score(selector_targets, selector_probabilities)),
        "pooled_oof_best_macro_f1": best_score,
        "pooled_oof_best_threshold": best_threshold,
        "pooled_oof_selected_samples": best_count,
        "nested_threshold_macro_f1": nested_score,
        "nested_thresholds": nested_thresholds,
        "recommendation": (
            "experimental_only" if nested_score <= base_score + 0.001 else "candidate"
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
