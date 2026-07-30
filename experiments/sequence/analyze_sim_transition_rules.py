import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from feature_utils_qwen_v4 import (
    ACTION_TO_FAMILY,
    ALL_CLASSES,
    FILE_PATH_PATTERN,
    FLAG_PATTERNS,
    GLOB_PATTERN,
    extract_state,
)


EXPLORE_CLASSES = ALL_CLASSES[:4]
EXPLORE_IDS = np.arange(4, dtype=np.int64)
LABEL2ID = {label: index for index, label in enumerate(ALL_CLASSES)}


def load_jsonl(path: Path) -> list[dict]:
    samples: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def softmax(values: np.ndarray) -> np.ndarray:
    values = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def fast_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 14,
) -> float:
    matrix = np.bincount(
        y_true * n_classes + y_pred,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)

    true_positive = np.diag(matrix).astype(np.float64)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)

    class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return float(class_f1.mean())


def session_and_step(sample_id: str) -> tuple[str, int]:
    match = re.match(r"(.+)-step_(\d+)$", sample_id)
    if match:
        return match.group(1), int(match.group(2))
    return sample_id, 0


def prompt_flags(prompt: str) -> dict[str, int]:
    prompt_lower = prompt.lower()

    values = {
        name: int(bool(re.search(pattern, prompt_lower)))
        for name, pattern in FLAG_PATTERNS.items()
    }
    values.update({
        "has_file_path": int(bool(FILE_PATH_PATTERN.search(prompt))),
        "has_glob": int(bool(GLOB_PATTERN.search(prompt))),
        "has_slash": int("/" in prompt or "\\" in prompt),
        "has_extension": int(
            bool(re.search(r"\.[a-zA-Z0-9]{1,8}\b", prompt))
        ),
        "has_question_mark": int("?" in prompt),
    })
    return values


def build_frame(
    samples: list[dict],
    label_map: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict] = []

    for index, sample in enumerate(samples):
        sample_id = str(sample["id"])
        session, step = session_and_step(sample_id)
        state = extract_state(sample)

        previous_actions = state["previous_actions"]
        last2 = (
            ">".join(previous_actions[-2:])
            if previous_actions
            else "none"
        )
        last3 = (
            ">".join(previous_actions[-3:])
            if previous_actions
            else "none"
        )

        last_item = (
            state["action_items"][-1]
            if state["action_items"]
            else {"result": "", "args": ""}
        )

        workspace = state["workspace"]
        prompt = state["current_prompt"]

        row = {
            "index": index,
            "id": sample_id,
            "session": session,
            "step": step,
            "source": (
                "sim"
                if sample_id.startswith("sess_sim_")
                else "au"
                if sample_id.startswith("sess_au_")
                else "other"
            ),
            "label": label_map[sample_id],
            "prompt": prompt,
            "last_action": state["last_action"],
            "second_last_action": state["second_last_action"],
            "last2": last2,
            "last3": last3,
            "history_len": state["history_len"],
            "action_count": state["action_count"],
            "failed_result_count": state["failed_result_count"],
            "last_result": last_item["result"],
            "last_args": last_item["args"],
            "open_file_count": len(state["open_files"]),
            "prompt_path_count": len(state["unique_paths"]),
            "ci": str(
                workspace.get("last_ci_status", "unknown")
            ).lower(),
            "git_dirty": str(
                workspace.get("git_dirty", "unknown")
            ).lower(),
            "top_language": state["top_language"],
        }
        row.update(prompt_flags(prompt))
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["y"] = frame["label"].map(LABEL2ID).astype(np.int64)
    return frame


def reconstruct_qwen(
    frame: pd.DataFrame,
    payload: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action_logits = payload["action_logits"].astype(np.float64)
    family_logits = payload["family_logits"].astype(np.float64)
    labels = payload["labels"].astype(np.int64)
    validation_indices = payload[
        "validation_indices"
    ].astype(np.int64)

    train_mask = np.ones(len(frame), dtype=bool)
    train_mask[validation_indices] = False

    training_counts = np.bincount(
        frame.loc[train_mask, "y"].to_numpy(),
        minlength=len(ALL_CLASSES),
    ).astype(np.float64)

    training_weights = (
        train_mask.sum()
        / (
            len(ALL_CLASSES)
            * np.maximum(training_counts, 1.0)
        )
    ) ** 0.25

    family_index = np.asarray(
        ACTION_TO_FAMILY,
        dtype=np.int64,
    )

    final_logits = (
        action_logits / 1.4
        + 1.25 * family_logits[:, family_index]
        + 0.85 * np.log(training_weights)[None, :]
    )

    predictions = final_logits.argmax(axis=1)
    probabilities = softmax(final_logits)

    if not np.array_equal(
        frame.iloc[validation_indices]["y"].to_numpy(),
        labels,
    ):
        raise RuntimeError(
            "validation_logits_v4.npz labels and train data do not align."
        )

    return (
        validation_indices,
        labels,
        final_logits,
        probabilities,
    )


def distribution_table(
    frame: pd.DataFrame,
    conditioning_columns: list[str],
) -> pd.DataFrame:
    subset = frame[
        (frame["source"] == "sim")
        & frame["label"].isin(EXPLORE_CLASSES)
    ].copy()

    rows: list[dict] = []

    group_argument: str | list[str]
    if len(conditioning_columns) == 1:
        group_argument = conditioning_columns[0]
    else:
        group_argument = conditioning_columns

    for state, group in subset.groupby(
        group_argument,
        dropna=False,
    ):
        if not isinstance(state, tuple):
            state = (state,)

        counts = group["label"].value_counts()
        total = len(group)

        row = {
            column: value
            for column, value in zip(
                conditioning_columns,
                state,
            )
        }
        row["count"] = total

        for label in EXPLORE_CLASSES:
            row[f"{label}_count"] = int(
                counts.get(label, 0)
            )
            row[f"{label}_rate"] = float(
                counts.get(label, 0) / total
            )

        row["majority_label"] = str(counts.index[0])
        row["majority_confidence"] = float(
            counts.iloc[0] / total
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["count", "majority_confidence"],
        ascending=[False, False],
    )


def majority_lookup(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    keys: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = train_frame[
        (train_frame["source"] == "sim")
        & (train_frame["y"] < 4)
    ]

    global_label = int(
        subset["y"].value_counts().index[0]
    )

    table: dict[tuple, tuple[int, int, float]] = {}

    group_argument: str | list[str]
    if len(keys) == 1:
        group_argument = keys[0]
    else:
        group_argument = keys

    for state, group in subset.groupby(
        group_argument,
        dropna=False,
    ):
        if not isinstance(state, tuple):
            state = (state,)

        counts = group["y"].value_counts()
        table[state] = (
            int(counts.index[0]),
            len(group),
            float(counts.iloc[0] / len(group)),
        )

    predictions = np.full(
        len(validation_frame),
        global_label,
        dtype=np.int64,
    )
    sample_counts = np.zeros(
        len(validation_frame),
        dtype=np.int64,
    )
    confidences = np.zeros(
        len(validation_frame),
        dtype=np.float64,
    )

    for row_index, state in enumerate(
        validation_frame[keys].itertuples(
            index=False,
            name=None,
        )
    ):
        if not isinstance(state, tuple):
            state = (state,)

        result = table.get(state)
        if result is not None:
            predictions[row_index] = result[0]
            sample_counts[row_index] = result[1]
            confidences[row_index] = result[2]

    return predictions, sample_counts, confidences


def evaluate_lookup_predictability(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    labels: np.ndarray,
    key_sets: list[list[str]],
) -> pd.DataFrame:
    true_sim_explore = (
        (validation_frame["source"].to_numpy() == "sim")
        & (labels < 4)
    )

    rows: list[dict] = []

    for keys in key_sets:
        prediction, counts, confidence = majority_lookup(
            train_frame,
            validation_frame,
            keys,
        )

        rows.append({
            "keys": "+".join(keys),
            "samples": int(true_sim_explore.sum()),
            "accuracy": float(
                np.mean(
                    prediction[true_sim_explore]
                    == labels[true_sim_explore]
                )
            ),
            "macro_f1": float(
                f1_score(
                    labels[true_sim_explore],
                    prediction[true_sim_explore],
                    labels=EXPLORE_IDS,
                    average="macro",
                    zero_division=0,
                )
            ),
            "coverage": float(
                np.mean(counts[true_sim_explore] > 0)
            ),
            "average_train_confidence": float(
                confidence[true_sim_explore].mean()
            ),
        })

    return pd.DataFrame(rows).sort_values(
        "macro_f1",
        ascending=False,
    )


def evaluate_rule_groups(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    labels: np.ndarray,
    qwen_predictions: np.ndarray,
    baseline_score: float,
    key_sets: list[list[str]],
) -> pd.DataFrame:
    rows: list[dict] = []
    sim_qwen_explore = (
        (validation_frame["source"].to_numpy() == "sim")
        & (qwen_predictions < 4)
    )

    for keys in key_sets:
        subset = train_frame[
            (train_frame["source"] == "sim")
            & (train_frame["y"] < 4)
        ]

        group_argument: str | list[str]
        if len(keys) == 1:
            group_argument = keys[0]
        else:
            group_argument = keys

        for state, group in subset.groupby(
            group_argument,
            dropna=False,
        ):
            if not isinstance(state, tuple):
                state = (state,)

            counts = group["y"].value_counts()
            majority_id = int(counts.index[0])
            train_count = len(group)
            train_confidence = float(
                counts.iloc[0] / train_count
            )

            if train_count < 50 or train_confidence < 0.45:
                continue

            condition = np.ones(
                len(validation_frame),
                dtype=bool,
            )
            for key, value in zip(keys, state):
                condition &= (
                    validation_frame[key].to_numpy()
                    == value
                )

            use = (
                condition
                & sim_qwen_explore
                & (qwen_predictions != majority_id)
            )

            if use.sum() < 3:
                continue

            changed_prediction = qwen_predictions.copy()
            changed_prediction[use] = majority_id

            gains = int(
                np.sum(
                    use
                    & (changed_prediction == labels)
                    & (qwen_predictions != labels)
                )
            )
            losses = int(
                np.sum(
                    use
                    & (changed_prediction != labels)
                    & (qwen_predictions == labels)
                )
            )
            score = fast_macro_f1(
                labels,
                changed_prediction,
            )

            rows.append({
                "key_type": "+".join(keys),
                "state": " | ".join(map(str, state)),
                "train_count": train_count,
                "train_majority": ALL_CLASSES[
                    majority_id
                ],
                "train_confidence": train_confidence,
                "override_count": int(use.sum()),
                "gains": gains,
                "losses": losses,
                "net_correct": gains - losses,
                "macro_f1": score,
                "macro_improvement": (
                    score - baseline_score
                ),
            })

    return pd.DataFrame(rows).sort_values(
        ["net_correct", "macro_improvement"],
        ascending=False,
    )


def candidate_rule_predictions(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    labels: np.ndarray,
    qwen_predictions: np.ndarray,
    qwen_probabilities: np.ndarray,
    key_sets: list[list[str]],
) -> tuple[pd.DataFrame, np.ndarray]:
    explore_probabilities = qwen_probabilities[:, :4]
    explore_probabilities /= explore_probabilities.sum(
        axis=1,
        keepdims=True,
    )
    sorted_explore = np.sort(
        explore_probabilities,
        axis=1,
    )
    explore_margin = (
        sorted_explore[:, -1]
        - sorted_explore[:, -2]
    )

    sim_qwen_explore = (
        (validation_frame["source"].to_numpy() == "sim")
        & (qwen_predictions < 4)
    )

    rows: list[dict] = []
    prediction_arrays: list[np.ndarray] = []

    for keys in key_sets:
        majority, counts, confidence = majority_lookup(
            train_frame,
            validation_frame,
            keys,
        )

        for min_confidence in (
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
        ):
            for max_margin in (
                0.05,
                0.10,
                0.15,
                0.20,
                0.30,
                1.00,
            ):
                use = (
                    sim_qwen_explore
                    & (counts >= 1)
                    & (confidence >= min_confidence)
                    & (majority != qwen_predictions)
                    & (explore_margin <= max_margin)
                )

                prediction = qwen_predictions.copy()
                prediction[use] = majority[use]
                score = fast_macro_f1(labels, prediction)

                rows.append({
                    "keys": "+".join(keys),
                    "min_train_confidence": min_confidence,
                    "max_qwen_explore_margin": max_margin,
                    "changed_samples": int(use.sum()),
                    "macro_f1": score,
                })
                prediction_arrays.append(
                    prediction.astype(np.int8)
                )

    result_frame = pd.DataFrame(rows)
    order = np.argsort(
        -result_frame["macro_f1"].to_numpy()
    )
    result_frame = result_frame.iloc[
        order
    ].reset_index(drop=True)
    prediction_matrix = np.stack(prediction_arrays)[
        order
    ]

    return result_frame, prediction_matrix


def nested_rule_evaluation(
    validation_frame: pd.DataFrame,
    labels: np.ndarray,
    qwen_predictions: np.ndarray,
    rule_frame: pd.DataFrame,
    prediction_matrix: np.ndarray,
) -> tuple[float, np.ndarray, list[dict]]:
    groups = validation_frame[
        "session"
    ].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=777,
    )

    nested_prediction = qwen_predictions.copy()
    selected: list[dict] = []

    for fold, (train_indices, held_indices) in enumerate(
        splitter.split(
            np.zeros(len(labels)),
            labels,
            groups,
        )
    ):
        scores = np.asarray([
            fast_macro_f1(
                labels[train_indices],
                prediction[train_indices],
            )
            for prediction in prediction_matrix
        ])
        best_index = int(scores.argmax())

        nested_prediction[held_indices] = (
            prediction_matrix[
                best_index,
                held_indices,
            ]
        )

        row = rule_frame.iloc[
            best_index
        ].to_dict()
        row.update({
            "fold": fold,
            "selection_macro_f1": float(
                scores[best_index]
            ),
            "held_samples": int(
                len(held_indices)
            ),
        })
        selected.append(row)

    return (
        fast_macro_f1(
            labels,
            nested_prediction,
        ),
        nested_prediction,
        selected,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/train.jsonl"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/train_labels.csv"),
    )
    parser.add_argument(
        "--logits",
        type=Path,
        default=Path("validation_logits_v4.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/sim_transition_rules"
        ),
    )
    args = parser.parse_args()

    started = time.perf_counter()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Load data...")
    samples = load_jsonl(args.data)
    label_frame = pd.read_csv(args.labels)
    label_map = dict(zip(
        label_frame["id"].astype(str),
        label_frame["action"].astype(str),
    ))

    print("Extract state transitions...")
    frame = build_frame(samples, label_map)

    payload = np.load(args.logits)
    (
        validation_indices,
        labels,
        final_logits,
        qwen_probabilities,
    ) = reconstruct_qwen(frame, payload)

    qwen_predictions = final_logits.argmax(axis=1)
    baseline_score = fast_macro_f1(
        labels,
        qwen_predictions,
    )

    train_mask = np.ones(
        len(frame),
        dtype=bool,
    )
    train_mask[validation_indices] = False

    train_frame = frame.loc[
        train_mask
    ].copy()
    validation_frame = frame.iloc[
        validation_indices
    ].reset_index(drop=True)

    print("Build transition tables...")
    second_last_table = distribution_table(
        frame,
        ["second_last_action"],
    )
    last_action_table = distribution_table(
        frame,
        ["last_action"],
    )
    step_table = distribution_table(
        frame,
        ["step"],
    )
    last_action_step_table = distribution_table(
        frame,
        ["last_action", "step"],
    )

    second_last_table.to_csv(
        args.output_dir
        / "sim_explore_second_last_transition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    last_action_table.to_csv(
        args.output_dir
        / "sim_explore_last_action_transition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    step_table.to_csv(
        args.output_dir
        / "sim_explore_step_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    last_action_step_table.to_csv(
        args.output_dir
        / "sim_explore_last_action_step.csv",
        index=False,
        encoding="utf-8-sig",
    )

    key_sets = [
        ["last_action"],
        ["second_last_action"],
        ["last2"],
        ["step"],
        ["last_action", "step"],
        ["last_action", "ci"],
        ["last_action", "open_file_count"],
        [
            "last_action",
            "has_open_intent",
            "has_search_intent",
            "has_directory_intent",
            "has_glob",
        ],
        [
            "step",
            "has_open_intent",
            "has_search_intent",
            "has_directory_intent",
            "has_glob",
        ],
    ]

    lookup_frame = evaluate_lookup_predictability(
        train_frame,
        validation_frame,
        labels,
        key_sets,
    )
    lookup_frame.to_csv(
        args.output_dir
        / "state_lookup_predictability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_rule_frame = evaluate_rule_groups(
        train_frame,
        validation_frame,
        labels,
        qwen_predictions,
        baseline_score,
        [
            ["second_last_action"],
            ["last_action", "step"],
            ["last2"],
            ["last_action", "open_file_count"],
        ],
    )
    group_rule_frame.to_csv(
        args.output_dir
        / "qwen_group_rule_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Evaluate Qwen override rules...")
    rule_frame, prediction_matrix = (
        candidate_rule_predictions(
            train_frame,
            validation_frame,
            labels,
            qwen_predictions,
            qwen_probabilities,
            [
                ["second_last_action"],
                ["last_action", "step"],
                ["last2"],
                ["last_action", "open_file_count"],
            ],
        )
    )
    rule_frame.to_csv(
        args.output_dir
        / "qwen_rule_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_prediction = prediction_matrix[0]
    pooled_score = float(
        rule_frame.iloc[0]["macro_f1"]
    )

    (
        nested_score,
        nested_prediction,
        nested_rules,
    ) = nested_rule_evaluation(
        validation_frame,
        labels,
        qwen_predictions,
        rule_frame,
        prediction_matrix,
    )

    # Fixed, interpretable exploration-chain rule:
    # list_directory(t-2) -> read_file(t)
    fixed_use = (
        (
            validation_frame[
                "source"
            ].to_numpy()
            == "sim"
        )
        & (qwen_predictions < 4)
        & (
            validation_frame[
                "second_last_action"
            ].to_numpy()
            == "list_directory"
        )
        & (
            qwen_predictions
            != LABEL2ID["read_file"]
        )
    )
    fixed_prediction = qwen_predictions.copy()
    fixed_prediction[
        fixed_use
    ] = LABEL2ID["read_file"]
    fixed_score = fast_macro_f1(
        labels,
        fixed_prediction,
    )

    groups = validation_frame[
        "session"
    ].astype(str).to_numpy()
    fixed_fold_results: list[dict] = []
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=777,
    )
    for fold, (_, held_indices) in enumerate(
        splitter.split(
            np.zeros(len(labels)),
            labels,
            groups,
        )
    ):
        baseline_fold = fast_macro_f1(
            labels[held_indices],
            qwen_predictions[held_indices],
        )
        fixed_fold = fast_macro_f1(
            labels[held_indices],
            fixed_prediction[held_indices],
        )
        fixed_fold_results.append({
            "fold": fold,
            "baseline_macro_f1": baseline_fold,
            "fixed_rule_macro_f1": fixed_fold,
            "improvement": (
                fixed_fold - baseline_fold
            ),
            "changed_samples": int(
                fixed_use[held_indices].sum()
            ),
        })

    source_rows: list[dict] = []
    for source in sorted(
        validation_frame["source"].unique()
    ):
        source_mask = (
            validation_frame[
                "source"
            ].to_numpy()
            == source
        )
        source_rows.append({
            "source": source,
            "samples": int(source_mask.sum()),
            "accuracy": float(
                np.mean(
                    qwen_predictions[source_mask]
                    == labels[source_mask]
                )
            ),
            "macro_f1_14class": float(
                f1_score(
                    labels[source_mask],
                    qwen_predictions[source_mask],
                    labels=np.arange(
                        len(ALL_CLASSES)
                    ),
                    average="macro",
                    zero_division=0,
                )
            ),
        })

        explore_mask = (
            source_mask
            & (labels < 4)
        )
        if explore_mask.any():
            source_rows[-1].update({
                "explore_samples": int(
                    explore_mask.sum()
                ),
                "explore_accuracy": float(
                    np.mean(
                        qwen_predictions[
                            explore_mask
                        ]
                        == labels[
                            explore_mask
                        ]
                    )
                ),
                "explore_macro_f1": float(
                    f1_score(
                        labels[explore_mask],
                        qwen_predictions[
                            explore_mask
                        ],
                        labels=EXPLORE_IDS,
                        average="macro",
                        zero_division=0,
                    )
                ),
            })

    source_frame = pd.DataFrame(
        source_rows
    )
    source_frame.to_csv(
        args.output_dir
        / "qwen_performance_by_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction_output = pd.DataFrame({
        "index": validation_indices,
        "id": validation_frame["id"],
        "true": [
            ALL_CLASSES[value]
            for value in labels
        ],
        "qwen": [
            ALL_CLASSES[value]
            for value in qwen_predictions
        ],
        "pooled_rule": [
            ALL_CLASSES[value]
            for value in pooled_prediction
        ],
        "nested_rule": [
            ALL_CLASSES[value]
            for value in nested_prediction
        ],
        "fixed_chain_rule": [
            ALL_CLASSES[value]
            for value in fixed_prediction
        ],
        "fixed_rule_applied": fixed_use,
    })
    prediction_output.to_csv(
        args.output_dir
        / "validation_rule_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Key transition rows used in the report.
    transition_lookup = (
        second_last_table.set_index(
            "second_last_action"
        )
    )

    chain_transitions: list[dict] = []
    for previous, expected in (
        ("list_directory", "read_file"),
        ("read_file", "grep_search"),
        ("grep_search", "glob_pattern"),
    ):
        if previous in transition_lookup.index:
            row = transition_lookup.loc[previous]
            chain_transitions.append({
                "second_last_action": previous,
                "expected_current_action": expected,
                "samples": int(row["count"]),
                "rate": float(
                    row[f"{expected}_rate"]
                ),
            })

    report = {
        "samples": len(frame),
        "sim_samples": int(
            (frame["source"] == "sim").sum()
        ),
        "au_samples": int(
            (frame["source"] == "au").sum()
        ),
        "validation_samples": len(labels),
        "qwen_macro_f1": baseline_score,
        "qwen_performance_by_source": (
            source_frame.to_dict(
                orient="records"
            )
        ),
        "exploration_chain": chain_transitions,
        "best_lookup_predictability": (
            lookup_frame.iloc[0].to_dict()
        ),
        "pooled_best_rule": (
            rule_frame.iloc[0].to_dict()
        ),
        "pooled_rule_improvement": (
            pooled_score - baseline_score
        ),
        "nested_rule_macro_f1": nested_score,
        "nested_rule_improvement": (
            nested_score - baseline_score
        ),
        "nested_selected_rules": nested_rules,
        "fixed_chain_rule": {
            "condition": (
                "source=sim AND "
                "qwen_family=explore AND "
                "second_last_action=list_directory "
                "AND qwen_prediction!=read_file"
            ),
            "changed_samples": int(
                fixed_use.sum()
            ),
            "macro_f1": fixed_score,
            "improvement": (
                fixed_score - baseline_score
            ),
            "fold_results": fixed_fold_results,
        },
        "recommendation": (
            "Keep the fixed list_directory(t-2)->"
            "read_file(t) rule as a low-risk candidate. "
            "Do not use the pooled generic rule without "
            "additional out-of-fold confirmation."
        ),
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }

    (
        args.output_dir / "report.json"
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    markdown = f"""# SIM Transition Rule Analysis

## Qwen baseline

- Validation Macro-F1: `{baseline_score:.6f}`
- Pooled best rule: `{pooled_score:.6f}`
- Nested rule estimate: `{nested_score:.6f}`
- Fixed chain rule: `{fixed_score:.6f}`

## Main finding

The synthetic `sim` sessions contain a strong two-turn exploration chain:

| Action at t-2 | Action at t | Rate |
|---|---|---:|
"""

    for item in chain_transitions:
        markdown += (
            f"| `{item['second_last_action']}` | "
            f"`{item['expected_current_action']}` | "
            f"{item['rate']:.2%} |\n"
        )

    markdown += f"""
Qwen already follows the `read_file -> grep_search` and
`grep_search -> glob_pattern` transitions in most validation
samples. The remaining actionable rule is:

```text
second_last_action == list_directory
AND source == sim
AND Qwen predicts an Explore action other than read_file
=> override to read_file
```

This changes `{int(fixed_use.sum())}` validation samples and moves
Macro-F1 from `{baseline_score:.6f}` to `{fixed_score:.6f}`.

## Interpretation

- `au` Explore performance is already high.
- `sim` Explore performance is the main bottleneck.
- One-step transition state is weak.
- Two-turn state is substantially more predictive.
- Generic threshold/grid rules overfit the single validation fold.
- The fixed exploration-chain rule is interpretable and more stable,
  but its gain is still small.
"""

    (
        args.output_dir / "report.md"
    ).write_text(
        markdown,
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
