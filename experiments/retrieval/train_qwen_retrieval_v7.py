from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
from scipy import sparse
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

from feature_utils_qwen_v4 import (
    ACTION_TO_FAMILY,
    ALL_CLASSES,
    build_structured_features,
    extract_state,
)


SEED = 42
NUM_CLASSES = len(ALL_CLASSES)
LABEL2ID = {
    label: index
    for index, label in enumerate(ALL_CLASSES)
}
FAMILY_INDEX = np.asarray(
    ACTION_TO_FAMILY,
    dtype=np.int64,
)

URL_RE = re.compile(
    r"(?:https?://|www\.)\S+",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-"
    r"[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-"
    r"[0-9a-f]{12}\b",
    re.IGNORECASE,
)
HASH_RE = re.compile(
    r"\b(?:[0-9a-f]{7,64})\b",
    re.IGNORECASE,
)
WINDOWS_PATH_RE = re.compile(
    r"\b[A-Za-z]:[\\/](?:[^\s<>:\"|?*]+[\\/])*"
    r"[^\s<>:\"|?*]*"
)
FILE_PATH_RE = re.compile(
    r"(?<![\w])(?:\.?\.?[/\\])?"
    r"(?:[\w.@+~:-]+[/\\])+"
    r"[\w.@+~:-]+(?:\.[A-Za-z0-9]{1,12})?"
)
FILE_NAME_RE = re.compile(
    r"\b[\w.@+~-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|"
    r"cpp|cc|c|h|hpp|cs|css|html|json|ya?ml|md|xml|"
    r"toml|ini|env|gradle|kt|swift|php|rb|sql|sh|ps1|"
    r"bat|vue|svelte|txt|csv)\b",
    re.IGNORECASE,
)
GLOB_RE = re.compile(
    r"(?:\*\*/|\*\.[A-Za-z0-9]+|"
    r"[?*][\w.-]*|\[[^\]]+\])"
)
NUMBER_RE = re.compile(
    r"(?<![\w])[-+]?\d+(?:\.\d+)?"
    r"(?:e[-+]?\d+)?(?![\w])",
    re.IGNORECASE,
)
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V7 retrieval-gated hybrid. It augments the V4 "
            "out-of-sample calibrator with state retrieval, "
            "exact-state lookup, and a dedicated Qwen-error selector."
        )
    )
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
        default=Path("model/qwen_retrieval_v7"),
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--selector-trees",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--selector-min-samples-leaf",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-features",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--word-max-features",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--char-max-features",
        type=int,
        default=60_000,
    )
    parser.add_argument(
        "--char-weight",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--disable-char",
        action="store_true",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compact_text(
    value: Any,
    limit: int = 1800,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    return WHITESPACE_RE.sub(
        " ",
        text,
    ).strip()[:limit]


def normalize_template(value: Any) -> str:
    text = compact_text(
        value,
        limit=2200,
    ).casefold()
    text = URL_RE.sub(" <URL> ", text)
    text = EMAIL_RE.sub(" <EMAIL> ", text)
    text = UUID_RE.sub(" <UUID> ", text)
    text = WINDOWS_PATH_RE.sub(" <PATH> ", text)
    text = FILE_PATH_RE.sub(" <PATH> ", text)
    text = FILE_NAME_RE.sub(" <FILE> ", text)
    text = GLOB_RE.sub(" <GLOB> ", text)
    text = HASH_RE.sub(" <HASH> ", text)
    text = NUMBER_RE.sub(" <NUM> ", text)
    return WHITESPACE_RE.sub(
        " ",
        text,
    ).strip()


def normalize_result(value: Any) -> str:
    text = normalize_template(value)
    text = re.sub(
        r"\b\d+\s*(?:matches?|files?|items?|lines?|"
        r"errors?|warnings?|tests?)\b",
        "<COUNT_RESULT>",
        text,
    )
    return WHITESPACE_RE.sub(
        " ",
        text,
    ).strip()


def source_name(sample: Mapping[str, Any]) -> str:
    sample_id = str(
        sample.get("id", "")
    )
    if sample_id.startswith("sess_sim_"):
        return "sim"
    if sample_id.startswith("sess_au_"):
        return "au"
    return "other"


def session_group(sample: Mapping[str, Any]) -> str:
    return str(
        sample.get("id", "")
    ).rsplit("-step_", 1)[0]


def canonical_json(
    value: Mapping[str, Any],
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def bucket_token(
    value: Any,
    boundaries: Sequence[int],
    prefix: str,
) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return f"__{prefix}_unknown__"

    previous = 0
    for boundary in boundaries:
        if number <= boundary:
            return (
                f"__{prefix}_{previous}_{boundary}__"
            )
        previous = boundary + 1

    return f"__{prefix}_{previous}_plus__"


def build_retrieval_text(
    sample: Mapping[str, Any],
) -> str:
    state = extract_state(dict(sample))
    meta = state["meta"]
    workspace = state["workspace"]
    actions = state["action_items"]
    sequence = state["previous_actions"][-6:]

    current_raw = compact_text(
        state["current_prompt"],
        limit=1400,
    ).casefold()
    current_template = normalize_template(
        state["current_prompt"]
    )

    parts: List[str] = [
        f"__current__ {current_raw}",
        f"__template__ {current_template}",
        f"__source_{source_name(sample)}__",
        f"__language_{state['top_language']}__",
        f"__ci_{str(workspace.get('last_ci_status', 'unknown')).casefold()}__",
        f"__git_{str(workspace.get('git_dirty', 'unknown')).casefold()}__",
        bucket_token(
            meta.get("turn_index"),
            [0, 1, 2, 4, 7, 12, 20, 40],
            "turn",
        ),
        bucket_token(
            state["action_count"],
            [0, 1, 2, 3, 5, 8, 12],
            "actions",
        ),
    ]

    if sequence:
        parts.extend([
            f"__last_{sequence[-1]}__",
            f"__last_{sequence[-1]}__",
            f"__last_{sequence[-1]}__",
        ])
        if len(sequence) >= 2:
            parts.extend([
                f"__second_{sequence[-2]}__",
                f"__second_{sequence[-2]}__",
            ])

        sequence_tokens = " ".join(
            f"__seq_{offset}_{action}__"
            for offset, action in enumerate(
                sequence
            )
        )
        parts.extend([
            sequence_tokens,
            sequence_tokens,
        ])

    for offset, user_text in enumerate(
        reversed(state["user_turns"][-2:]),
        start=1,
    ):
        parts.append(
            f"__prev_user_{offset}__ "
            f"{normalize_template(user_text)[:500]}"
        )

    for offset, item in enumerate(
        reversed(actions[-2:]),
        start=1,
    ):
        parts.extend([
            f"__prev_action_{offset}_{item['name']}__",
            f"__prev_args_{offset}__ "
            f"{normalize_template(item.get('args', ''))[:350]}",
            f"__prev_result_{offset}__ "
            f"{normalize_result(item.get('result', ''))[:450]}",
        ])

    return "\n".join(parts)


def make_exact_keys(
    sample: Mapping[str, Any],
) -> Dict[str, str]:
    state = extract_state(dict(sample))
    template = normalize_template(
        state["current_prompt"]
    )
    raw_prompt = compact_text(
        state["current_prompt"],
        limit=2200,
    ).casefold()
    source = source_name(sample)
    sequence = tuple(
        state["previous_actions"][-5:]
    )

    return {
        "raw_prompt": canonical_json({
            "source": source,
            "prompt": raw_prompt,
        }),
        "normalized_template": canonical_json({
            "source": source,
            "prompt": template,
        }),
        "template_last_action": canonical_json({
            "source": source,
            "prompt": template,
            "last": state["last_action"],
        }),
        "template_action_context": canonical_json({
            "source": source,
            "prompt": template,
            "last": state["last_action"],
            "second": state["second_last_action"],
            "sequence": sequence,
        }),
    }


def load_dataset(
    data_path: Path,
    labels_path: Path,
) -> Tuple[List[dict], np.ndarray]:
    label_map: Dict[str, int] = {}

    with labels_path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        required = {"id", "action"}
        if not required.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                "train_labels.csv requires id and action columns."
            )

        for row in reader:
            action = str(row["action"])
            if action not in LABEL2ID:
                raise ValueError(
                    f"Unknown action: {action}"
                )
            label_map[str(row["id"])] = (
                LABEL2ID[action]
            )

    samples: List[dict] = []
    labels: List[int] = []

    with data_path.open(
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            sample = json.loads(line)
            sample_id = str(
                sample.get("id", "")
            )
            if sample_id not in label_map:
                raise KeyError(
                    f"Missing label for id={sample_id}, "
                    f"line={line_number}"
                )

            samples.append(sample)
            labels.append(
                label_map[sample_id]
            )

    return (
        samples,
        np.asarray(
            labels,
            dtype=np.int64,
        ),
    )


def reconstruct_v4_logits(
    payload: np.lib.npyio.NpzFile,
    all_labels: np.ndarray,
    validation_indices: np.ndarray,
) -> np.ndarray:
    train_mask = np.ones(
        len(all_labels),
        dtype=bool,
    )
    train_mask[validation_indices] = False

    counts = np.bincount(
        all_labels[train_mask],
        minlength=NUM_CLASSES,
    ).astype(np.float64)

    training_class_weights = np.power(
        train_mask.sum()
        / (
            NUM_CLASSES
            * np.maximum(counts, 1.0)
        ),
        0.25,
    )

    action_logits = payload[
        "action_logits"
    ].astype(np.float64)
    family_logits = payload[
        "family_logits"
    ].astype(np.float64)

    return (
        action_logits / 1.4
        + 1.25
        * family_logits[:, FAMILY_INDEX]
        + 0.85
        * np.log(
            np.maximum(
                training_class_weights,
                1e-12,
            )
        )[None, :]
    )


def build_qwen_features(
    samples: Sequence[dict],
    action_logits: np.ndarray,
    family_logits: np.ndarray,
    final_logits: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    probabilities = softmax(
        final_logits,
        axis=1,
    )
    centered_logits = (
        final_logits
        - final_logits.mean(
            axis=1,
            keepdims=True,
        )
    )
    sorted_probabilities = np.sort(
        probabilities,
        axis=1,
    )

    margin = (
        sorted_probabilities[:, -1]
        - sorted_probabilities[:, -2]
    )[:, None]
    maximum_probability = (
        probabilities.max(
            axis=1,
            keepdims=True,
        )
    )
    entropy = (
        -np.sum(
            probabilities
            * np.log(
                np.maximum(
                    probabilities,
                    1e-12,
                )
            ),
            axis=1,
        )
    )[:, None]

    structured = np.stack([
        build_structured_features(sample)
        for sample in samples
    ]).astype(np.float32)

    predicted_ids = final_logits.argmax(
        axis=1
    )
    predicted_one_hot = np.eye(
        NUM_CLASSES,
        dtype=np.float32,
    )[predicted_ids]

    source_sim = np.asarray([
        float(
            source_name(sample) == "sim"
        )
        for sample in samples
    ], dtype=np.float32)[:, None]

    features = np.hstack([
        action_logits.astype(np.float32),
        family_logits.astype(np.float32),
        final_logits.astype(np.float32),
        centered_logits.astype(np.float32),
        probabilities.astype(np.float32),
        predicted_one_hot,
        structured,
        margin.astype(np.float32),
        maximum_probability.astype(np.float32),
        entropy.astype(np.float32),
        source_sim,
    ]).astype(np.float32)

    names: List[str] = []
    names.extend(
        f"action_logit_{label}"
        for label in ALL_CLASSES
    )
    names.extend(
        f"family_logit_{index}"
        for index in range(
            family_logits.shape[1]
        )
    )
    names.extend(
        f"final_logit_{label}"
        for label in ALL_CLASSES
    )
    names.extend(
        f"centered_logit_{label}"
        for label in ALL_CLASSES
    )
    names.extend(
        f"probability_{label}"
        for label in ALL_CLASSES
    )
    names.extend(
        f"qwen_prediction_{label}"
        for label in ALL_CLASSES
    )
    names.extend(
        f"structured_{index}"
        for index in range(
            structured.shape[1]
        )
    )
    names.extend([
        "qwen_margin",
        "qwen_max_probability",
        "qwen_entropy",
        "source_sim",
    ])

    return features, names


def balanced_sample_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    maximum: int,
    seed: int,
) -> np.ndarray:
    if len(indices) <= maximum:
        return indices

    rng = np.random.default_rng(seed)
    selected: List[int] = []
    per_class = max(
        1,
        maximum // NUM_CLASSES,
    )

    for class_id in range(NUM_CLASSES):
        candidates = indices[
            labels[indices] == class_id
        ].copy()
        rng.shuffle(candidates)
        selected.extend(
            candidates[:per_class].tolist()
        )

    selected_array = np.asarray(
        selected,
        dtype=np.int64,
    )

    if len(selected_array) < maximum:
        selected_set = set(
            selected_array.tolist()
        )
        remaining = np.asarray([
            index
            for index in indices
            if int(index) not in selected_set
        ], dtype=np.int64)
        rng.shuffle(remaining)
        selected_array = np.concatenate([
            selected_array,
            remaining[
                : maximum - len(selected_array)
            ],
        ])

    rng.shuffle(selected_array)
    return selected_array[:maximum]


def fit_retrieval_index(
    train_texts: Sequence[str],
    validation_texts: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[
    TfidfVectorizer,
    TfidfVectorizer | None,
    sparse.csr_matrix,
    sparse.csr_matrix,
]:
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9995,
        max_features=args.word_max_features,
        sublinear_tf=True,
        lowercase=False,
        dtype=np.float32,
        token_pattern=(
            r"(?u)\b\w[\w./<>:=+@-]*\b"
        ),
    )

    print("Fit retrieval Word TF-IDF...")
    word_train = (
        word_vectorizer.fit_transform(
            train_texts
        )
    )
    word_validation = (
        word_vectorizer.transform(
            validation_texts
        )
    )

    char_vectorizer = None

    if args.disable_char:
        train_matrix = word_train
        validation_matrix = word_validation
    else:
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_df=0.9995,
            max_features=args.char_max_features,
            sublinear_tf=True,
            lowercase=False,
            dtype=np.float32,
        )

        print("Fit retrieval Char TF-IDF...")
        char_train = (
            char_vectorizer.fit_transform(
                train_texts
            )
        )
        char_validation = (
            char_vectorizer.transform(
                validation_texts
            )
        )

        train_matrix = sparse.hstack([
            word_train,
            char_train * args.char_weight,
        ], format="csr", dtype=np.float32)
        validation_matrix = sparse.hstack([
            word_validation,
            char_validation
            * args.char_weight,
        ], format="csr", dtype=np.float32)

    train_matrix = normalize(
        train_matrix,
        norm="l2",
        copy=False,
    ).tocsr()
    validation_matrix = normalize(
        validation_matrix,
        norm="l2",
        copy=False,
    ).tocsr()

    return (
        word_vectorizer,
        char_vectorizer,
        train_matrix,
        validation_matrix,
    )


def retrieve_neighbors(
    train_matrix: sparse.csr_matrix,
    validation_matrix: sparse.csr_matrix,
    maximum_neighbors: int,
    query_batch_size: int,
    n_jobs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    neighbor_count = min(
        maximum_neighbors,
        train_matrix.shape[0],
    )
    if neighbor_count <= 0:
        raise RuntimeError(
            "No retrieval training samples."
        )

    model = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=n_jobs,
    )
    model.fit(train_matrix)

    distance_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []

    for start in tqdm(
        range(
            0,
            validation_matrix.shape[0],
            query_batch_size,
        ),
        desc="Retrieve SIM neighbors",
    ):
        end = min(
            validation_matrix.shape[0],
            start + query_batch_size,
        )
        distances, indices = model.kneighbors(
            validation_matrix[start:end],
            return_distance=True,
        )
        distance_parts.append(
            distances.astype(
                np.float32,
            )
        )
        index_parts.append(
            indices.astype(
                np.int32,
            )
        )

    return (
        1.0 - np.vstack(distance_parts),
        np.vstack(index_parts),
    )


def probability_entropy(
    probabilities: np.ndarray,
) -> np.ndarray:
    return -np.sum(
        probabilities
        * np.log(
            np.maximum(
                probabilities,
                1e-12,
            )
        ),
        axis=1,
    )


def build_retrieval_features(
    validation_size: int,
    sim_validation_positions: np.ndarray,
    neighbor_similarities: np.ndarray,
    neighbor_labels: np.ndarray,
    qwen_predictions: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    k_values = [
        value
        for value in (1, 3, 5, 9, 15, 25)
        if value
        <= neighbor_similarities.shape[1]
    ]
    powers = (2.0, 8.0)

    local_parts: List[np.ndarray] = []
    names: List[str] = []

    for k_value in k_values:
        similarities = (
            neighbor_similarities[:, :k_value]
        )
        labels = neighbor_labels[:, :k_value]

        for power in powers:
            weights = np.power(
                np.maximum(
                    similarities,
                    1e-6,
                ),
                power,
            )
            probabilities = np.stack([
                np.sum(
                    weights
                    * (labels == class_id),
                    axis=1,
                )
                for class_id in range(
                    NUM_CLASSES
                )
            ], axis=1)
            probabilities /= probabilities.sum(
                axis=1,
                keepdims=True,
            ).clip(min=1e-12)

            local_parts.append(
                probabilities.astype(
                    np.float32,
                )
            )
            names.extend(
                f"retrieval_k{k_value}_p{int(power)}_"
                f"probability_{label}"
                for label in ALL_CLASSES
            )

        default_weights = np.power(
            np.maximum(
                similarities,
                1e-6,
            ),
            4.0,
        )
        default_probabilities = np.stack([
            np.sum(
                default_weights
                * (labels == class_id),
                axis=1,
            )
            for class_id in range(
                NUM_CLASSES
            )
        ], axis=1)
        default_probabilities /= (
            default_probabilities.sum(
                axis=1,
                keepdims=True,
            ).clip(min=1e-12)
        )
        retrieval_prediction = (
            default_probabilities.argmax(
                axis=1
            )
        )

        summary = np.column_stack([
            similarities[:, 0],
            similarities.mean(axis=1),
            default_probabilities.max(axis=1),
            probability_entropy(
                default_probabilities
            ),
            (
                retrieval_prediction
                == qwen_predictions[
                    sim_validation_positions
                ]
            ).astype(np.float32),
            (
                FAMILY_INDEX[
                    retrieval_prediction
                ]
                == FAMILY_INDEX[
                    qwen_predictions[
                        sim_validation_positions
                    ]
                ]
            ).astype(np.float32),
        ]).astype(np.float32)

        local_parts.append(summary)
        names.extend([
            f"retrieval_k{k_value}_top_similarity",
            f"retrieval_k{k_value}_mean_similarity",
            f"retrieval_k{k_value}_consensus",
            f"retrieval_k{k_value}_entropy",
            f"retrieval_k{k_value}_agrees_qwen",
            f"retrieval_k{k_value}_family_agrees_qwen",
        ])

    top_count = min(
        8,
        neighbor_similarities.shape[1],
    )
    local_parts.append(
        neighbor_similarities[
            :, :top_count
        ].astype(np.float32)
    )
    names.extend(
        f"retrieval_top_similarity_{index + 1}"
        for index in range(top_count)
    )

    local_features = np.hstack(
        local_parts
    ).astype(np.float32)

    features = np.zeros(
        (
            validation_size,
            local_features.shape[1] + 1,
        ),
        dtype=np.float32,
    )
    features[
        sim_validation_positions,
        :-1,
    ] = local_features
    features[
        sim_validation_positions,
        -1,
    ] = 1.0
    names.append("retrieval_source_available")

    return features, names


def build_exact_lookup_features(
    samples: Sequence[dict],
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    qwen_predictions: np.ndarray,
) -> Tuple[
    np.ndarray,
    List[str],
    Dict[str, Dict[str, Dict[str, Any]]],
]:
    level_names = [
        "raw_prompt",
        "normalized_template",
        "template_last_action",
        "template_action_context",
    ]

    counters: Dict[
        str,
        Dict[str, Counter[int]],
    ] = {
        level: defaultdict(Counter)
        for level in level_names
    }

    print("Build exact-state lookup tables...")
    for index in tqdm(
        train_indices,
        desc="Exact lookup train",
    ):
        keys = make_exact_keys(
            samples[int(index)]
        )
        label = int(labels[int(index)])

        for level in level_names:
            counters[level][
                keys[level]
            ][label] += 1

    width_per_level = NUM_CLASSES + 4
    features = np.zeros(
        (
            len(validation_indices),
            width_per_level
            * len(level_names),
        ),
        dtype=np.float32,
    )
    names: List[str] = []

    for level_offset, level in enumerate(
        level_names
    ):
        base = (
            level_offset
            * width_per_level
        )
        names.extend(
            f"lookup_{level}_prediction_{label}"
            for label in ALL_CLASSES
        )
        names.extend([
            f"lookup_{level}_hit",
            f"lookup_{level}_confidence",
            f"lookup_{level}_log_count",
            f"lookup_{level}_disagrees_qwen",
        ])

        for row, dataset_index in enumerate(
            tqdm(
                validation_indices,
                desc=f"Exact lookup {level}",
                leave=False,
            )
        ):
            key = make_exact_keys(
                samples[int(dataset_index)]
            )[level]
            counter = counters[level].get(
                key
            )
            if not counter:
                continue

            prediction, maximum_count = max(
                counter.items(),
                key=lambda item: (
                    item[1],
                    -item[0],
                ),
            )
            total_count = sum(
                counter.values()
            )
            confidence = (
                maximum_count
                / total_count
            )

            features[
                row,
                base + prediction,
            ] = 1.0
            features[
                row,
                base + NUM_CLASSES,
            ] = 1.0
            features[
                row,
                base + NUM_CLASSES + 1,
            ] = float(confidence)
            features[
                row,
                base + NUM_CLASSES + 2,
            ] = min(
                1.0,
                math.log1p(total_count)
                / math.log1p(100.0),
            )
            features[
                row,
                base + NUM_CLASSES + 3,
            ] = float(
                prediction
                != qwen_predictions[row]
            )

    serializable_lookup: Dict[
        str,
        Dict[str, Dict[str, Any]],
    ] = {}

    for level in level_names:
        serializable_lookup[level] = {
            key: {
                "counts": {
                    str(label): int(count)
                    for label, count
                    in counter.items()
                },
                "total": int(
                    sum(counter.values())
                ),
            }
            for key, counter
            in counters[level].items()
        }

    return (
        features,
        names,
        serializable_lookup,
    )


def aligned_predict_proba(
    model: ExtraTreesClassifier,
    features: np.ndarray,
    class_count: int,
) -> np.ndarray:
    local = model.predict_proba(
        features
    )
    result = np.full(
        (
            len(features),
            class_count,
        ),
        1e-9,
        dtype=np.float64,
    )
    result[
        :,
        model.classes_.astype(int),
    ] = local
    result /= result.sum(
        axis=1,
        keepdims=True,
    )
    return result


def macro_f1_fast(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> float:
    confusion = np.bincount(
        labels * NUM_CLASSES
        + predictions,
        minlength=NUM_CLASSES
        * NUM_CLASSES,
    ).reshape(
        NUM_CLASSES,
        NUM_CLASSES,
    )
    true_positive = np.diag(
        confusion
    ).astype(np.float64)
    denominator = (
        confusion.sum(axis=0)
        + confusion.sum(axis=1)
    ).astype(np.float64)
    class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros(
            NUM_CLASSES,
            dtype=np.float64,
        ),
        where=denominator > 0,
    )
    return float(
        class_f1.mean()
    )


def candidate_configs() -> List[dict]:
    configs: List[dict] = []

    for alpha in (
        0.4,
        0.6,
        0.8,
        1.0,
    ):
        for selector_threshold in (
            0.45,
            0.55,
            0.65,
            0.75,
        ):
            for alternative_confidence in (
                0.45,
                0.55,
                0.65,
                0.75,
            ):
                for maximum_qwen_margin in (
                    0.05,
                    0.10,
                    0.20,
                    0.40,
                ):
                    for mode in (
                        "all",
                        "sim_only",
                        "same_family",
                        "sim_same_family",
                    ):
                        configs.append({
                            "alpha": alpha,
                            "selector_threshold": (
                                selector_threshold
                            ),
                            "alternative_confidence": (
                                alternative_confidence
                            ),
                            "maximum_qwen_margin": (
                                maximum_qwen_margin
                            ),
                            "mode": mode,
                        })

    return configs


def apply_candidate(
    qwen_probabilities: np.ndarray,
    alternative_probabilities: np.ndarray,
    selector_probabilities: np.ndarray,
    source_sim: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    qwen_prediction = (
        qwen_probabilities.argmax(
            axis=1
        )
    )
    alternative_prediction = (
        alternative_probabilities.argmax(
            axis=1
        )
    )
    sorted_qwen = np.sort(
        qwen_probabilities,
        axis=1,
    )
    qwen_margin = (
        sorted_qwen[:, -1]
        - sorted_qwen[:, -2]
    )
    alternative_confidence = (
        alternative_probabilities.max(
            axis=1
        )
    )

    alpha = float(
        config["alpha"]
    )
    blended_log_probability = (
        (1.0 - alpha)
        * np.log(
            np.maximum(
                qwen_probabilities,
                1e-12,
            )
        )
        + alpha
        * np.log(
            np.maximum(
                alternative_probabilities,
                1e-12,
            )
        )
    )
    blended_prediction = (
        blended_log_probability.argmax(
            axis=1
        )
    )

    gate = (
        selector_probabilities
        >= float(
            config[
                "selector_threshold"
            ]
        )
    ) & (
        alternative_confidence
        >= float(
            config[
                "alternative_confidence"
            ]
        )
    ) & (
        qwen_margin
        <= float(
            config[
                "maximum_qwen_margin"
            ]
        )
    ) & (
        alternative_prediction
        != qwen_prediction
    )

    mode = str(config["mode"])
    same_family = (
        FAMILY_INDEX[
            alternative_prediction
        ]
        == FAMILY_INDEX[
            qwen_prediction
        ]
    )

    if mode == "sim_only":
        gate &= source_sim
    elif mode == "same_family":
        gate &= same_family
    elif mode == "sim_same_family":
        gate &= (
            source_sim
            & same_family
        )

    result = qwen_prediction.copy()
    result[gate] = (
        blended_prediction[gate]
    )
    return result


def build_candidate_prediction_matrix(
    qwen_probabilities: np.ndarray,
    alternative_probabilities: np.ndarray,
    selector_probabilities: np.ndarray,
    source_sim: np.ndarray,
    configs: Sequence[dict],
) -> np.ndarray:
    matrix = np.empty(
        (
            len(configs),
            len(qwen_probabilities),
        ),
        dtype=np.uint8,
    )

    for index, config in enumerate(
        tqdm(
            configs,
            desc="Evaluate V7 gates",
        )
    ):
        matrix[index] = apply_candidate(
            qwen_probabilities,
            alternative_probabilities,
            selector_probabilities,
            source_sim,
            config,
        ).astype(np.uint8)

    return matrix


def choose_best_candidate(
    labels: np.ndarray,
    prediction_matrix: np.ndarray,
    indices: np.ndarray,
) -> Tuple[int, float]:
    best_index = 0
    best_score = -1.0

    local_labels = labels[indices]

    for candidate_index in range(
        prediction_matrix.shape[0]
    ):
        score = macro_f1_fast(
            local_labels,
            prediction_matrix[
                candidate_index,
                indices,
            ].astype(np.int64),
        )
        if score > best_score:
            best_score = score
            best_index = candidate_index

    return best_index, best_score


def subset_score(
    labels: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> float:
    if not np.any(mask):
        return 0.0

    return float(
        f1_score(
            labels[mask],
            predictions[mask],
            labels=np.arange(
                NUM_CLASSES
            ),
            average="macro",
            zero_division=0,
        )
    )


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    started = time.perf_counter()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Load data...")
    samples, all_labels = load_dataset(
        args.data,
        args.labels,
    )
    payload = np.load(
        args.logits
    )
    validation_indices_full = payload[
        "validation_indices"
    ].astype(np.int64)
    validation_labels_full = payload[
        "labels"
    ].astype(np.int64)

    if not np.array_equal(
        all_labels[
            validation_indices_full
        ],
        validation_labels_full,
    ):
        raise RuntimeError(
            "validation_logits_v4.npz does not "
            "align with train data."
        )

    train_mask = np.ones(
        len(samples),
        dtype=bool,
    )
    train_mask[
        validation_indices_full
    ] = False
    train_indices = np.flatnonzero(
        train_mask
    )
    validation_positions = np.arange(
        len(validation_indices_full),
        dtype=np.int64,
    )

    if args.smoke:
        train_indices = (
            balanced_sample_indices(
                train_indices,
                all_labels,
                maximum=6000,
                seed=SEED,
            )
        )
        validation_positions = (
            balanced_sample_indices(
                validation_positions,
                validation_labels_full,
                maximum=2000,
                seed=SEED + 1,
            )
        )
        args.trees = min(
            args.trees,
            100,
        )
        args.selector_trees = min(
            args.selector_trees,
            100,
        )
        args.word_max_features = min(
            args.word_max_features,
            25_000,
        )
        args.char_max_features = min(
            args.char_max_features,
            15_000,
        )
        args.max_neighbors = min(
            args.max_neighbors,
            16,
        )
        args.disable_char = True
        print(
            "Smoke subset:",
            f"train={len(train_indices)}",
            f"validation={len(validation_positions)}",
        )

    validation_indices = (
        validation_indices_full[
            validation_positions
        ]
    )
    validation_labels = (
        validation_labels_full[
            validation_positions
        ]
    )
    validation_samples = [
        samples[int(index)]
        for index in validation_indices
    ]

    all_final_logits = (
        reconstruct_v4_logits(
            payload,
            all_labels,
            validation_indices_full,
        )
    )
    action_logits = payload[
        "action_logits"
    ][validation_positions]
    family_logits = payload[
        "family_logits"
    ][validation_positions]
    final_logits = all_final_logits[
        validation_positions
    ]
    qwen_probabilities = softmax(
        final_logits,
        axis=1,
    )
    qwen_predictions = (
        final_logits.argmax(
            axis=1
        )
    )
    qwen_score = macro_f1_fast(
        validation_labels,
        qwen_predictions,
    )

    print(
        f"Reproduced Qwen V4 Macro-F1: "
        f"{qwen_score:.6f}"
    )

    qwen_features, qwen_names = (
        build_qwen_features(
            validation_samples,
            action_logits,
            family_logits,
            final_logits,
        )
    )

    source_sim = np.asarray([
        source_name(sample) == "sim"
        for sample in validation_samples
    ], dtype=bool)

    sim_train_indices = np.asarray([
        index
        for index in train_indices
        if source_name(
            samples[int(index)]
        ) == "sim"
    ], dtype=np.int64)
    sim_validation_positions = (
        np.flatnonzero(
            source_sim
        )
    )

    if (
        len(sim_train_indices) == 0
        or len(sim_validation_positions) == 0
    ):
        raise RuntimeError(
            "SIM retrieval samples are missing."
        )

    print(
        "Build retrieval texts:",
        f"train_sim={len(sim_train_indices)}",
        f"validation_sim={len(sim_validation_positions)}",
    )
    train_retrieval_texts = [
        build_retrieval_text(
            samples[int(index)]
        )
        for index in tqdm(
            sim_train_indices,
            desc="Retrieval train text",
        )
    ]
    validation_retrieval_texts = [
        build_retrieval_text(
            validation_samples[
                int(position)
            ]
        )
        for position in tqdm(
            sim_validation_positions,
            desc="Retrieval validation text",
        )
    ]

    (
        word_vectorizer,
        char_vectorizer,
        retrieval_train_matrix,
        retrieval_validation_matrix,
    ) = fit_retrieval_index(
        train_retrieval_texts,
        validation_retrieval_texts,
        args,
    )

    print(
        "Retrieval feature matrix:",
        f"train={retrieval_train_matrix.shape}",
        f"validation={retrieval_validation_matrix.shape}",
        f"train_nnz={retrieval_train_matrix.nnz:,}",
    )

    (
        neighbor_similarities,
        neighbor_indices,
    ) = retrieve_neighbors(
        retrieval_train_matrix,
        retrieval_validation_matrix,
        args.max_neighbors,
        args.query_batch_size,
        args.n_jobs,
    )
    neighbor_labels = all_labels[
        sim_train_indices[
            neighbor_indices
        ]
    ]

    (
        retrieval_features,
        retrieval_names,
    ) = build_retrieval_features(
        len(validation_samples),
        sim_validation_positions,
        neighbor_similarities,
        neighbor_labels,
        qwen_predictions,
    )

    (
        exact_features,
        exact_names,
        exact_lookup,
    ) = build_exact_lookup_features(
        samples,
        all_labels,
        train_indices,
        validation_indices,
        qwen_predictions,
    )

    features = np.hstack([
        qwen_features,
        retrieval_features,
        exact_features,
    ]).astype(np.float32)
    feature_names = (
        qwen_names
        + retrieval_names
        + exact_names
    )

    print(
        "V7 feature matrix:",
        features.shape,
    )

    groups = np.asarray([
        session_group(sample)
        for sample in validation_samples
    ])
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=777,
    )
    splits = list(
        splitter.split(
            features,
            validation_labels,
            groups,
        )
    )

    alternative_oof = np.zeros(
        (
            len(validation_labels),
            NUM_CLASSES,
        ),
        dtype=np.float64,
    )
    selector_oof = np.zeros(
        len(validation_labels),
        dtype=np.float64,
    )
    fold_ids = np.full(
        len(validation_labels),
        -1,
        dtype=np.int64,
    )

    qwen_error_target = (
        validation_labels
        != qwen_predictions
    ).astype(np.int64)

    print("Train grouped OOF V7 models...")
    for fold, (
        fit_indices,
        held_indices,
    ) in enumerate(splits):
        alternative_model = (
            ExtraTreesClassifier(
                n_estimators=args.trees,
                min_samples_leaf=(
                    args.min_samples_leaf
                ),
                max_features=(
                    args.max_features
                ),
                class_weight="balanced",
                n_jobs=args.n_jobs,
                random_state=SEED + fold,
            )
        )
        selector_model = (
            ExtraTreesClassifier(
                n_estimators=(
                    args.selector_trees
                ),
                min_samples_leaf=(
                    args.selector_min_samples_leaf
                ),
                max_features=(
                    args.max_features
                ),
                class_weight="balanced",
                n_jobs=args.n_jobs,
                random_state=(
                    SEED + 100 + fold
                ),
            )
        )

        alternative_model.fit(
            features[fit_indices],
            validation_labels[
                fit_indices
            ],
        )
        selector_model.fit(
            features[fit_indices],
            qwen_error_target[
                fit_indices
            ],
        )

        alternative_oof[
            held_indices
        ] = aligned_predict_proba(
            alternative_model,
            features[held_indices],
            NUM_CLASSES,
        )
        selector_probabilities = (
            selector_model.predict_proba(
                features[held_indices]
            )
        )
        positive_column = int(
            np.flatnonzero(
                selector_model.classes_
                == 1
            )[0]
        )
        selector_oof[
            held_indices
        ] = selector_probabilities[
            :,
            positive_column,
        ]
        fold_ids[held_indices] = fold

        print(
            f"  fold={fold} "
            f"train={len(fit_indices)} "
            f"held={len(held_indices)}"
        )

    alternative_prediction = (
        alternative_oof.argmax(
            axis=1
        )
    )
    alternative_score = macro_f1_fast(
        validation_labels,
        alternative_prediction,
    )

    try:
        selector_auc = float(
            roc_auc_score(
                qwen_error_target,
                selector_oof,
            )
        )
    except ValueError:
        selector_auc = 0.0

    configs = candidate_configs()
    prediction_matrix = (
        build_candidate_prediction_matrix(
            qwen_probabilities,
            alternative_oof,
            selector_oof,
            source_sim,
            configs,
        )
    )

    all_positions = np.arange(
        len(validation_labels),
        dtype=np.int64,
    )
    (
        pooled_config_index,
        pooled_score,
    ) = choose_best_candidate(
        validation_labels,
        prediction_matrix,
        all_positions,
    )
    pooled_prediction = (
        prediction_matrix[
            pooled_config_index
        ].astype(np.int64)
    )
    pooled_config = configs[
        pooled_config_index
    ]

    nested_prediction = (
        qwen_predictions.copy()
    )
    nested_configs: List[dict] = []

    for fold in range(5):
        held_indices = np.flatnonzero(
            fold_ids == fold
        )
        selection_indices = np.flatnonzero(
            fold_ids != fold
        )
        (
            best_index,
            selection_score,
        ) = choose_best_candidate(
            validation_labels,
            prediction_matrix,
            selection_indices,
        )
        nested_prediction[
            held_indices
        ] = prediction_matrix[
            best_index,
            held_indices,
        ]
        nested_configs.append({
            "fold": fold,
            "selection_macro_f1": float(
                selection_score
            ),
            "held_samples": int(
                len(held_indices)
            ),
            **configs[best_index],
        })

    nested_score = macro_f1_fast(
        validation_labels,
        nested_prediction,
    )

    sim_mask = source_sim
    au_mask = np.asarray([
        source_name(sample) == "au"
        for sample in validation_samples
    ], dtype=bool)

    metrics = {
        "qwen_macro_f1": qwen_score,
        "alternative_only_oof_macro_f1": (
            alternative_score
        ),
        "selector_auc": selector_auc,
        "pooled_v7_macro_f1": pooled_score,
        "pooled_improvement": (
            pooled_score - qwen_score
        ),
        "pooled_changed_samples": int(
            np.sum(
                pooled_prediction
                != qwen_predictions
            )
        ),
        "pooled_config": pooled_config,
        "nested_v7_macro_f1": nested_score,
        "nested_improvement": (
            nested_score - qwen_score
        ),
        "nested_changed_samples": int(
            np.sum(
                nested_prediction
                != qwen_predictions
            )
        ),
        "qwen_sim_macro_f1": subset_score(
            validation_labels,
            qwen_predictions,
            sim_mask,
        ),
        "v7_sim_macro_f1": subset_score(
            validation_labels,
            nested_prediction,
            sim_mask,
        ),
        "qwen_au_macro_f1": subset_score(
            validation_labels,
            qwen_predictions,
            au_mask,
        ),
        "v7_au_macro_f1": subset_score(
            validation_labels,
            nested_prediction,
            au_mask,
        ),
        "feature_count": int(
            features.shape[1]
        ),
        "retrieval_train_samples": int(
            len(sim_train_indices)
        ),
        "retrieval_validation_samples": int(
            len(sim_validation_positions)
        ),
        "nested_configs": nested_configs,
    }

    print(
        f"Alternative-only OOF Macro-F1: "
        f"{alternative_score:.6f}"
    )
    print(
        f"Selector OOF AUC: "
        f"{selector_auc:.6f}"
    )
    print(
        f"Pooled V7 Macro-F1: "
        f"{pooled_score:.6f}"
    )
    print(
        f"Nested V7 Macro-F1: "
        f"{nested_score:.6f}"
    )
    print(
        f"Nested improvement: "
        f"{nested_score - qwen_score:+.6f}"
    )
    print(
        "SIM:",
        f"{metrics['qwen_sim_macro_f1']:.6f}",
        "->",
        f"{metrics['v7_sim_macro_f1']:.6f}",
    )
    print(
        "AU:",
        f"{metrics['qwen_au_macro_f1']:.6f}",
        "->",
        f"{metrics['v7_au_macro_f1']:.6f}",
    )

    report_text = classification_report(
        validation_labels,
        nested_prediction,
        labels=np.arange(
            NUM_CLASSES
        ),
        target_names=ALL_CLASSES,
        digits=6,
        zero_division=0,
    )
    print(report_text)

    print("Train final V7 models...")
    final_alternative_model = (
        ExtraTreesClassifier(
            n_estimators=args.trees,
            min_samples_leaf=(
                args.min_samples_leaf
            ),
            max_features=(
                args.max_features
            ),
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=SEED,
        )
    )
    final_selector_model = (
        ExtraTreesClassifier(
            n_estimators=args.selector_trees,
            min_samples_leaf=(
                args.selector_min_samples_leaf
            ),
            max_features=(
                args.max_features
            ),
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=SEED + 100,
        )
    )
    final_alternative_model.fit(
        features,
        validation_labels,
    )
    final_selector_model.fit(
        features,
        qwen_error_target,
    )

    bundle = {
        "version": "qwen_retrieval_v7",
        "classes": ALL_CLASSES,
        "action_to_family": ACTION_TO_FAMILY,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "char_weight": args.char_weight,
        "retrieval_train_matrix": (
            retrieval_train_matrix
        ),
        "retrieval_train_labels": (
            all_labels[
                sim_train_indices
            ].astype(np.int8)
        ),
        "exact_lookup": exact_lookup,
        "alternative_model": (
            final_alternative_model
        ),
        "selector_model": (
            final_selector_model
        ),
        "feature_names": feature_names,
        "pooled_config": pooled_config,
        "nested_configs": nested_configs,
        "max_neighbors": int(
            args.max_neighbors
        ),
        "retrieval_k_values": [
            value
            for value in (
                1,
                3,
                5,
                9,
                15,
                25,
            )
            if value
            <= neighbor_similarities.shape[1]
        ],
        "retrieval_powers": [2.0, 8.0],
    }

    print("Save V7 bundle...")
    joblib.dump(
        bundle,
        args.output_dir
        / "retrieval_calibrator.joblib",
        compress=3,
    )

    metrics["elapsed_seconds"] = float(
        time.perf_counter() - started
    )
    (
        args.output_dir / "report.json"
    ).write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (
        args.output_dir
        / "classification_report.txt"
    ).write_text(
        report_text,
        encoding="utf-8",
    )

    np.savez_compressed(
        args.output_dir
        / "validation_outputs_v7.npz",
        validation_indices=(
            validation_indices
        ),
        labels=validation_labels,
        qwen_predictions=(
            qwen_predictions
        ),
        alternative_oof=(
            alternative_oof.astype(
                np.float32
            )
        ),
        selector_oof=(
            selector_oof.astype(
                np.float32
            )
        ),
        pooled_predictions=(
            pooled_prediction
        ),
        nested_predictions=(
            nested_prediction
        ),
        fold_ids=fold_ids,
        neighbor_similarities=(
            neighbor_similarities.astype(
                np.float32
            )
        ),
        neighbor_labels=(
            neighbor_labels.astype(
                np.int8
            )
        ),
    )

    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
