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
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score

from feature_utils_qwen_v4 import (
    ACTION_TO_FAMILY,
    ALL_CLASSES,
    FAMILY_NAMES,
    build_structured_features,
    extract_state,
)


SEED = 42
LABEL2ID = {label: index for index, label in enumerate(ALL_CLASSES)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
NUM_CLASSES = len(ALL_CLASSES)
NUM_FAMILIES = len(FAMILY_NAMES)

# Qwen V4와 동일한 family 정의를 사용한다.
FAMILY_TO_ACTIONS: Dict[int, List[int]] = {
    family_id: [
        action_id
        for action_id, mapped_family in enumerate(ACTION_TO_FAMILY)
        if mapped_family == family_id
    ]
    for family_id in range(NUM_FAMILIES)
}

WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
HASH_RE = re.compile(r"\b(?:[0-9a-f]{7,64})\b", re.IGNORECASE)
WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:[\\/](?:[^\s<>:\"|?*]+[\\/])*[^\s<>:\"|?*]*")
FILE_PATH_RE = re.compile(
    r"(?<![\w])(?:\.?\.?[/\\])?(?:[\w.@+~:-]+[/\\])+[\w.@+~:-]+(?:\.[A-Za-z0-9]{1,12})?"
)
FILE_NAME_RE = re.compile(
    r"\b[\w.@+~-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cpp|cc|c|h|hpp|cs|css|html|json|ya?ml|md|xml|toml|ini|env|gradle|kt|swift|php|rb|sql|sh|ps1|bat|vue|svelte|txt|csv)\b",
    re.IGNORECASE,
)
GLOB_RE = re.compile(r"(?:\*\*/|\*\.[A-Za-z0-9]+|[?*][\w.-]*|\[[^\]]+\])")
NUMBER_RE = re.compile(r"(?<![\w])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![\w])", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Word/Char TF-IDF + structured state + family router + specialists + "
            "template override 모델을 V4와 동일 Fold에서 학습·검증합니다."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument(
        "--fold-indices",
        type=Path,
        default=Path("analysis/state_conflicts/fold_0_indices.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("model/hierarchical_tfidf_v1"))
    parser.add_argument("--smoke", action="store_true", help="클래스당 일부 샘플로 파이프라인 점검")
    parser.add_argument("--smoke-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument("--word-max-features", type=int, default=140_000)
    parser.add_argument("--char-max-features", type=int, default=180_000)
    parser.add_argument("--word-min-df", type=int, default=2)
    parser.add_argument("--char-min-df", type=int, default=2)
    parser.add_argument("--structured-scale", type=float, default=2.0)

    parser.add_argument("--model", choices=("linearsvc", "logreg", "sgd"), default="linearsvc")
    parser.add_argument("--c", type=float, default=4.0)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--class-weight-exp", type=float, default=0.25)
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compact_text(value: Any, limit: int = 2400) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return WHITESPACE_RE.sub(" ", text).strip()[:limit]


def normalize_template(value: Any) -> str:
    text = compact_text(value).casefold()
    text = URL_RE.sub(" <URL> ", text)
    text = EMAIL_RE.sub(" <EMAIL> ", text)
    text = UUID_RE.sub(" <UUID> ", text)
    text = WINDOWS_PATH_RE.sub(" <PATH> ", text)
    text = FILE_PATH_RE.sub(" <PATH> ", text)
    text = FILE_NAME_RE.sub(" <FILE> ", text)
    text = GLOB_RE.sub(" <GLOB> ", text)
    text = HASH_RE.sub(" <HASH> ", text)
    text = NUMBER_RE.sub(" <NUM> ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def source_from_id(sample_id: str) -> str:
    if sample_id.startswith("sess_sim_"):
        return "sim"
    if sample_id.startswith("sess_au_"):
        return "au"
    return "other"


def path_extensions(paths: Sequence[Any]) -> List[str]:
    result: List[str] = []
    for raw_path in paths:
        name = str(raw_path).replace("\\", "/").rsplit("/", 1)[-1].casefold()
        if "." in name and not name.startswith("."):
            result.append(name.rsplit(".", 1)[-1])
        elif name.startswith("."):
            result.append(name)
        else:
            result.append("noext")
    return sorted(set(result))


def bucket_number(value: Any, edges: Sequence[float], prefix: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{prefix}_unknown"
    for index, edge in enumerate(edges):
        if number <= edge:
            return f"{prefix}_{index}"
    return f"{prefix}_{len(edges)}"


def build_model_text(sample: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Returns:
        word_text: current prompt + prefixed state/history tokens
        char_text: current prompt 중심 문자 n-gram 입력
        lookup_key: normalized prompt + last action
    """
    state = extract_state(sample)
    meta = state["meta"]
    workspace = state["workspace"]
    actions = state["action_items"]

    current_raw = compact_text(state["current_prompt"], 2200).casefold()
    current_template = normalize_template(current_raw)
    last_action = state["last_action"]
    second_last_action = state["second_last_action"]
    sequence = state["previous_actions"][-5:]

    parts: List[str] = [
        f"__current__ {current_raw}",
        f"__template__ {current_template}",
        # 중요한 상태 토큰은 여러 번 넣어 선형 모델에서 충분한 가중치를 갖게 한다.
        f"__last_action_{last_action}__ __last_action_{last_action}__",
        f"__second_last_action_{second_last_action}__",
        "__action_sequence__ " + " ".join(f"__seq_{i}_{name}__" for i, name in enumerate(sequence)),
        f"__source_{source_from_id(state['sample_id'])}__",
        f"__ci_{str(workspace.get('last_ci_status', 'unknown')).casefold()}__",
        f"__git_dirty_{str(workspace.get('git_dirty', 'unknown')).casefold()}__",
        f"__language_{state['top_language']}__",
        f"__lang_pref_{str(meta.get('language_pref', 'unknown')).casefold()}__",
        bucket_number(meta.get("turn_index"), [0, 1, 2, 4, 7, 12, 20, 40], "turn"),
        bucket_number(state["history_len"], [0, 2, 4, 6, 8, 12, 20], "history"),
        bucket_number(state["action_count"], [0, 1, 2, 3, 5, 8, 12], "actions"),
        bucket_number(len(state["open_files"]), [0, 1, 2, 4, 7, 10], "openfiles"),
    ]

    extensions = path_extensions(state["open_files"])
    if extensions:
        parts.append(" ".join(f"__open_ext_{ext}__" for ext in extensions))

    for offset, user_text in enumerate(reversed(state["user_turns"][-3:]), start=1):
        parts.append(f"__prev_user_{offset}__ {normalize_template(user_text)}")

    for offset, item in enumerate(reversed(actions[-3:]), start=1):
        parts.extend([
            f"__prev_action_{offset}_{item['name']}__",
            f"__prev_args_{offset}__ {normalize_template(item.get('args', ''))}",
            f"__prev_result_{offset}__ {normalize_template(item.get('result', ''))}",
        ])

    word_text = "\n".join(parts)
    char_text = "\n".join([
        current_raw,
        current_template,
        f"last={last_action} second={second_last_action}",
    ])
    lookup_key = json.dumps(
        {"p": current_template, "last": last_action},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return word_text, char_text, lookup_key


def load_dataset(data_path: Path, labels_path: Path) -> Tuple[List[dict], np.ndarray]:
    label_map: Dict[str, int] = {}
    with labels_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not {"id", "action"}.issubset(reader.fieldnames or []):
            raise ValueError("train_labels.csv에는 id, action 컬럼이 필요합니다.")
        for row in reader:
            action = str(row["action"])
            if action not in LABEL2ID:
                raise ValueError(f"알 수 없는 action: {action}")
            label_map[str(row["id"])] = LABEL2ID[action]

    samples: List[dict] = []
    labels: List[int] = []
    with data_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            sample_id = str(sample.get("id", ""))
            if sample_id not in label_map:
                raise KeyError(f"라벨이 없는 sample id: {sample_id}, line={line_number}")
            samples.append(sample)
            labels.append(label_map[sample_id])

    if len(samples) != len(label_map):
        raise RuntimeError(f"샘플/라벨 수 불일치: samples={len(samples)}, labels={len(label_map)}")
    return samples, np.asarray(labels, dtype=np.int64)


def load_fold_indices(path: Path, samples: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    required = {"index", "id", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"fold index 파일 필수 컬럼: {sorted(required)}")
    frame = frame.sort_values("index")
    if len(frame) != len(samples):
        raise RuntimeError(f"fold rows={len(frame)} != samples={len(samples)}")

    expected_ids = [str(sample.get("id", "")) for sample in samples]
    actual_ids = frame["id"].astype(str).tolist()
    if expected_ids != actual_ids:
        mismatch = next((i for i, (a, b) in enumerate(zip(expected_ids, actual_ids)) if a != b), None)
        raise RuntimeError(f"fold index의 샘플 순서가 train.jsonl과 다릅니다. first mismatch={mismatch}")

    split = frame["split"].astype(str).to_numpy()
    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "validation")
    if len(train_idx) + len(val_idx) != len(samples):
        raise RuntimeError("split 값은 train 또는 validation이어야 합니다.")
    return train_idx, val_idx


def balanced_smoke_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    for class_id in range(NUM_CLASSES):
        candidates = np.flatnonzero(labels == class_id)
        rng.shuffle(candidates)
        chosen.extend(candidates[:per_class].tolist())
    chosen_array = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(chosen_array)
    return chosen_array


def class_weight_map(labels: np.ndarray, classes: Sequence[int], exponent: float) -> Dict[int, float]:
    counts = Counter(int(value) for value in labels)
    n = max(1, len(labels))
    k = max(1, len(classes))
    return {
        int(class_id): float((n / (k * max(1, counts[int(class_id)]))) ** exponent)
        for class_id in classes
    }


def create_classifier(
    model_kind: str,
    labels: np.ndarray,
    classes: Sequence[int],
    class_weight_exp: float,
    c_value: float,
    max_iter: int,
    tol: float,
    n_jobs: int,
    seed: int,
):
    weights = class_weight_map(labels, classes, class_weight_exp)
    if model_kind == "sgd":
        # alpha는 LogisticRegression의 C와 완전히 같지는 않지만 C가 커질수록 규제를 줄인다.
        alpha = 1.0 / max(1.0, c_value * max(1, len(labels)))
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            class_weight=weights,
            random_state=seed,
            n_jobs=n_jobs,
            average=True,
        )

    if model_kind == "linearsvc":
        return LinearSVC(
            C=c_value,
            class_weight=weights,
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
            dual="auto",
        )

    # saga는 sparse multinomial을 지원하고 scikit-learn 1.8+에서도 동작한다.
    return LogisticRegression(
        C=c_value,
        solver="saga",
        max_iter=max_iter,
        tol=tol,
        class_weight=weights,
        random_state=seed,
    )


def model_probabilities(model, x_matrix, temperature: float = 1.0) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_matrix).astype(np.float64, copy=False)

    scores = np.asarray(model.decision_function(x_matrix), dtype=np.float64)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores = scores / max(1e-6, float(temperature))
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True).clip(min=1e-12)


def safe_predict_proba(model, x_matrix, num_columns: int, class_ids: Sequence[int] | None = None) -> np.ndarray:
    raw = model_probabilities(model, x_matrix)
    output = np.zeros((x_matrix.shape[0], num_columns), dtype=np.float64)
    model_classes = np.asarray(model.classes_, dtype=np.int64)
    output[:, model_classes] = raw
    return output


def build_lookup(keys: Sequence[str], labels: np.ndarray) -> Dict[str, Tuple[int, int, float]]:
    counters: Dict[str, Counter] = defaultdict(Counter)
    for key, label in zip(keys, labels):
        counters[key][int(label)] += 1

    lookup: Dict[str, Tuple[int, int, float]] = {}
    for key, counts in counters.items():
        label, best_count = counts.most_common(1)[0]
        total = sum(counts.values())
        lookup[key] = (int(label), int(total), float(best_count / total))
    return lookup


def apply_lookup_override(
    predictions: np.ndarray,
    keys: Sequence[str],
    lookup: Mapping[str, Tuple[int, int, float]],
    min_count: int,
    min_confidence: float,
) -> Tuple[np.ndarray, int, float]:
    result = predictions.copy()
    accepted = 0
    for index, key in enumerate(keys):
        item = lookup.get(key)
        if item is None:
            continue
        label, count, confidence = item
        if count >= min_count and confidence >= min_confidence:
            result[index] = label
            accepted += 1
    return result, accepted, accepted / max(1, len(keys))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        )
    )


def tune_blend(
    y_true: np.ndarray,
    global_probs: np.ndarray,
    hierarchical_probs: np.ndarray,
    structured_probs: np.ndarray,
    class_priors: np.ndarray,
    val_lookup_keys: Sequence[str],
    lookup: Mapping[str, Tuple[int, int, float]],
) -> Tuple[Dict[str, float], np.ndarray, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    best_score = -1.0
    best_config: Dict[str, float] = {}
    best_predictions: np.ndarray | None = None

    for alpha_global in (0.25, 0.40, 0.55, 0.70, 0.85, 1.00):
        base = alpha_global * global_probs + (1.0 - alpha_global) * hierarchical_probs
        for beta_structured in (0.0, 0.05, 0.10, 0.20, 0.30):
            blended = (1.0 - beta_structured) * base + beta_structured * structured_probs
            for prior_beta in (-0.20, -0.10, 0.0, 0.10, 0.20, 0.35):
                adjusted = blended / np.power(class_priors[None, :].clip(min=1e-8), prior_beta)
                predictions = adjusted.argmax(axis=1)
                score = macro_f1(y_true, predictions)
                rows.append({
                    "alpha_global": alpha_global,
                    "beta_structured": beta_structured,
                    "prior_beta": prior_beta,
                    "template_min_count": 0,
                    "template_min_confidence": 0.0,
                    "template_coverage": 0.0,
                    "macro_f1": score,
                })
                if score > best_score:
                    best_score = score
                    best_config = {
                        "alpha_global": alpha_global,
                        "beta_structured": beta_structured,
                        "prior_beta": prior_beta,
                        "template_min_count": 0,
                        "template_min_confidence": 0.0,
                    }
                    best_predictions = predictions.copy()

                for min_count in (2, 3, 5, 10):
                    for min_confidence in (0.95, 0.98, 1.0):
                        overridden, accepted, coverage = apply_lookup_override(
                            predictions,
                            val_lookup_keys,
                            lookup,
                            min_count=min_count,
                            min_confidence=min_confidence,
                        )
                        override_score = macro_f1(y_true, overridden)
                        rows.append({
                            "alpha_global": alpha_global,
                            "beta_structured": beta_structured,
                            "prior_beta": prior_beta,
                            "template_min_count": min_count,
                            "template_min_confidence": min_confidence,
                            "template_coverage": coverage,
                            "macro_f1": override_score,
                        })
                        if override_score > best_score:
                            best_score = override_score
                            best_config = {
                                "alpha_global": alpha_global,
                                "beta_structured": beta_structured,
                                "prior_beta": prior_beta,
                                "template_min_count": min_count,
                                "template_min_confidence": min_confidence,
                                "template_accepted": accepted,
                                "template_coverage": coverage,
                            }
                            best_predictions = overridden.copy()

    assert best_predictions is not None
    best_config["macro_f1"] = best_score
    result_frame = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return best_config, best_predictions, result_frame


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Load data...")
    samples, labels = load_dataset(args.data, args.labels)
    train_idx, val_idx = load_fold_indices(args.fold_indices, samples)

    if args.smoke:
        selected = balanced_smoke_indices(labels, args.smoke_per_class, args.seed)
        selected_set = set(selected.tolist())
        train_idx = np.asarray([i for i in train_idx if i in selected_set], dtype=np.int64)
        val_idx = np.asarray([i for i in val_idx if i in selected_set], dtype=np.int64)
        print(f"Smoke subset: train={len(train_idx)} val={len(val_idx)}")
    else:
        print(f"Full data: train={len(train_idx)} val={len(val_idx)}")

    print("Build text/state features...")
    word_texts: List[str] = []
    char_texts: List[str] = []
    lookup_keys: List[str] = []
    structured_rows: List[np.ndarray] = []

    active_indices = np.concatenate([train_idx, val_idx])
    active_set = set(active_indices.tolist())
    # smoke 모드에서는 불필요한 70k 전체 특징 생성을 피한다.
    for index, sample in enumerate(samples):
        if index not in active_set:
            word_texts.append("")
            char_texts.append("")
            lookup_keys.append("")
            structured_rows.append(np.empty(0, dtype=np.float32))
            continue
        word_text, char_text, lookup_key = build_model_text(sample)
        word_texts.append(word_text)
        char_texts.append(char_text)
        lookup_keys.append(lookup_key)
        structured_rows.append(build_structured_features(sample))

    train_word = [word_texts[i] for i in train_idx]
    val_word = [word_texts[i] for i in val_idx]
    train_char = [char_texts[i] for i in train_idx]
    val_char = [char_texts[i] for i in val_idx]

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=args.word_min_df,
        max_df=0.9995,
        max_features=args.word_max_features,
        sublinear_tf=True,
        lowercase=False,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w[\w./<>:=+@-]*\b",
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=args.char_min_df,
        max_features=args.char_max_features,
        sublinear_tf=True,
        lowercase=False,
        dtype=np.float32,
    )

    print("Fit Word TF-IDF...")
    x_word_train = word_vectorizer.fit_transform(train_word)
    x_word_val = word_vectorizer.transform(val_word)
    print("Fit Char TF-IDF...")
    x_char_train = char_vectorizer.fit_transform(train_char)
    x_char_val = char_vectorizer.transform(val_char)

    x_struct_train = sparse.csr_matrix(
        np.stack([structured_rows[i] for i in train_idx]).astype(np.float32) * args.structured_scale
    )
    x_struct_val = sparse.csr_matrix(
        np.stack([structured_rows[i] for i in val_idx]).astype(np.float32) * args.structured_scale
    )

    x_train = sparse.hstack([x_word_train, x_char_train, x_struct_train], format="csr", dtype=np.float32)
    x_val = sparse.hstack([x_word_val, x_char_val, x_struct_val], format="csr", dtype=np.float32)
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    family_train = np.asarray([ACTION_TO_FAMILY[int(label)] for label in y_train], dtype=np.int64)

    print(
        f"Feature matrix: train={x_train.shape}, val={x_val.shape}, "
        f"nnz={x_train.nnz:,}"
    )

    print("Train global 14-class model...")
    global_model = create_classifier(
        args.model, y_train, list(range(NUM_CLASSES)), args.class_weight_exp,
        args.c, args.max_iter, args.tol, args.n_jobs, args.seed,
    )
    global_model.fit(x_train, y_train)
    global_probs = safe_predict_proba(global_model, x_val, NUM_CLASSES)
    global_pred = global_probs.argmax(axis=1)
    global_score = macro_f1(y_val, global_pred)
    print(f"Global Macro-F1: {global_score:.6f}")

    print("Train 5-class family router...")
    router_model = create_classifier(
        args.model, family_train, list(range(NUM_FAMILIES)), args.class_weight_exp,
        args.c, args.max_iter, args.tol, args.n_jobs, args.seed + 1,
    )
    router_model.fit(x_train, family_train)
    router_probs = safe_predict_proba(router_model, x_val, NUM_FAMILIES)

    print("Train family specialists...")
    specialist_models: Dict[int, Any] = {}
    conditional_probs = np.zeros((len(val_idx), NUM_CLASSES), dtype=np.float64)
    for family_id, action_ids in FAMILY_TO_ACTIONS.items():
        if len(action_ids) == 1:
            conditional_probs[:, action_ids[0]] = 1.0
            print(f"  {FAMILY_NAMES[family_id]}: one-class passthrough -> {ALL_CLASSES[action_ids[0]]}")
            continue
        mask = family_train == family_id
        specialist_y = y_train[mask]
        specialist = create_classifier(
            args.model,
            specialist_y,
            action_ids,
            args.class_weight_exp,
            args.c,
            args.max_iter,
            args.tol,
            args.n_jobs,
            args.seed + 10 + family_id,
        )
        specialist.fit(x_train[mask], specialist_y)
        specialist_models[family_id] = specialist
        probs = model_probabilities(specialist, x_val)
        for local_column, action_id in enumerate(specialist.classes_.astype(int)):
            conditional_probs[:, action_id] = probs[:, local_column]
        print(f"  {FAMILY_NAMES[family_id]}: train={int(mask.sum())}, classes={[ALL_CLASSES[i] for i in action_ids]}")

    hierarchical_probs = np.zeros_like(global_probs)
    for action_id, family_id in enumerate(ACTION_TO_FAMILY):
        hierarchical_probs[:, action_id] = router_probs[:, family_id] * conditional_probs[:, action_id]
    hierarchical_probs /= hierarchical_probs.sum(axis=1, keepdims=True).clip(min=1e-12)
    hierarchical_score = macro_f1(y_val, hierarchical_probs.argmax(axis=1))
    print(f"Hierarchical Macro-F1: {hierarchical_score:.6f}")

    print("Train structured-only model...")
    structured_model = create_classifier(
        args.model, y_train, list(range(NUM_CLASSES)), args.class_weight_exp,
        args.c, args.max_iter, args.tol, args.n_jobs, args.seed + 2,
    )
    structured_model.fit(x_struct_train, y_train)
    structured_probs = safe_predict_proba(structured_model, x_struct_val, NUM_CLASSES)
    structured_score = macro_f1(y_val, structured_probs.argmax(axis=1))
    print(f"Structured-only Macro-F1: {structured_score:.6f}")

    train_lookup_keys = [lookup_keys[i] for i in train_idx]
    val_lookup_keys = [lookup_keys[i] for i in val_idx]
    lookup = build_lookup(train_lookup_keys, y_train)
    priors = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    priors /= priors.sum()

    print("Tune probability blend and template override...")
    best_config, best_predictions, blend_frame = tune_blend(
        y_val,
        global_probs,
        hierarchical_probs,
        structured_probs,
        priors,
        val_lookup_keys,
        lookup,
    )
    best_score = float(best_config["macro_f1"])
    print("Best configuration:")
    print(json.dumps(best_config, ensure_ascii=False, indent=2))

    report_text = classification_report(
        y_val,
        best_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=ALL_CLASSES,
        digits=6,
        zero_division=0,
    )
    print(report_text)

    prediction_frame = pd.DataFrame({
        "index": val_idx,
        "id": [str(samples[i].get("id", "")) for i in val_idx],
        "true_action": [ID2LABEL[int(value)] for value in y_val],
        "pred_action": [ID2LABEL[int(value)] for value in best_predictions],
        "correct": y_val == best_predictions,
        "global_pred": [ID2LABEL[int(value)] for value in global_pred],
    })
    prediction_frame.to_csv(args.output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    blend_frame.head(500).to_csv(args.output_dir / "blend_search_top500.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    metadata = {
        "seed": args.seed,
        "smoke": args.smoke,
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "feature_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "word_features": int(len(word_vectorizer.vocabulary_)),
        "char_features": int(len(char_vectorizer.vocabulary_)),
        "structured_features": int(x_struct_train.shape[1]),
        "model_kind": args.model,
        "c": args.c,
        "class_weight_exp": args.class_weight_exp,
        "scores": {
            "global_macro_f1": global_score,
            "hierarchical_macro_f1": hierarchical_score,
            "structured_macro_f1": structured_score,
            "best_macro_f1": best_score,
        },
        "best_config": best_config,
        "classes": ALL_CLASSES,
        "families": FAMILY_NAMES,
        "action_to_family": ACTION_TO_FAMILY,
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.output_dir / "metadata.json", metadata)

    if not args.no_save_model:
        print("Save model bundle...")
        bundle = {
            "word_vectorizer": word_vectorizer,
            "char_vectorizer": char_vectorizer,
            "global_model": global_model,
            "router_model": router_model,
            "specialist_models": specialist_models,
            "structured_model": structured_model,
            "template_lookup": lookup,
            "class_priors": priors,
            "best_config": best_config,
            "structured_scale": args.structured_scale,
            "classes": ALL_CLASSES,
            "families": FAMILY_NAMES,
            "action_to_family": ACTION_TO_FAMILY,
        }
        joblib.dump(bundle, args.output_dir / "model.joblib", compress=3)

    elapsed = time.perf_counter() - started
    print(f"Saved: {args.output_dir}")
    print(f"Best validation Macro-F1: {best_score:.6f}")
    print(f"Elapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
