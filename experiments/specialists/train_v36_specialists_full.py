from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train full-data V33/V35 specialist artifacts for submission inference."
    )
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/v36_specialists_full"),
    )
    parser.add_argument("--word-features", type=int, default=70000)
    parser.add_argument("--char-features", type=int, default=70000)
    parser.add_argument("--min-df", type=int, default=2)
    return parser.parse_args()


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


def load_data(data_path: Path, labels_path: Path) -> Tuple[List[str], np.ndarray]:
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
    texts = [build_text(sample) for sample in samples]
    return texts, labels


def train_pair(
    all_texts: List[str],
    all_labels: np.ndarray,
    pair: Tuple[int, int],
    c_value: float,
    word_features: int,
    char_features: int,
    min_df: int,
    output_dir: Path,
    name: str,
) -> dict:
    mask = np.isin(all_labels, pair)
    texts = [text for text, keep in zip(all_texts, mask) if keep]
    labels = all_labels[mask]
    binary_labels = (labels == pair[1]).astype(np.int64)

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

    x_word = word.fit_transform(texts)
    x_char = char.fit_transform(texts)
    x_train = hstack([x_word, x_char], format="csr")

    model = LogisticRegression(
        C=c_value,
        max_iter=1500,
        solver="liblinear",
        class_weight="balanced",
        random_state=SEED,
    )
    model.fit(x_train, binary_labels)

    pair_dir = output_dir / name
    pair_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(word, pair_dir / "word_vectorizer.joblib", compress=3)
    joblib.dump(char, pair_dir / "char_vectorizer.joblib", compress=3)
    joblib.dump(model, pair_dir / "model.joblib", compress=3)

    metadata = {
        "name": name,
        "class_ids": list(pair),
        "classes": [CLASSES[pair[0]], CLASSES[pair[1]]],
        "positive_class_id": pair[1],
        "positive_class": CLASSES[pair[1]],
        "train_samples": int(mask.sum()),
        "class_counts": {
            CLASSES[pair[0]]: int((labels == pair[0]).sum()),
            CLASSES[pair[1]]: int((labels == pair[1]).sum()),
        },
        "C": c_value,
        "word_features_actual": int(len(word.vocabulary_)),
        "char_features_actual": int(len(char.vocabulary_)),
        "min_df": min_df,
    }
    (pair_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    texts, labels = load_data(args.data, args.labels)

    print("=" * 80)
    print("TRAIN FULL V36 SPECIALISTS")
    print("=" * 80)
    print(f"Total samples: {len(texts)}")

    read_list = train_pair(
        texts,
        labels,
        pair=(READ_ID, LIST_ID),
        c_value=1.0,
        word_features=args.word_features,
        char_features=args.char_features,
        min_df=args.min_df,
        output_dir=args.output_dir,
        name="read_list",
    )
    print(
        f"read_list: samples={read_list['train_samples']} "
        f"word={read_list['word_features_actual']} "
        f"char={read_list['char_features_actual']}"
    )

    ask_plan = train_pair(
        texts,
        labels,
        pair=(ASK_ID, PLAN_ID),
        c_value=0.25,
        word_features=args.word_features,
        char_features=args.char_features,
        min_df=args.min_df,
        output_dir=args.output_dir,
        name="ask_plan",
    )
    print(
        f"ask_plan: samples={ask_plan['train_samples']} "
        f"word={ask_plan['word_features_actual']} "
        f"char={ask_plan['char_features_actual']}"
    )

    config = {
        "version": "v36_v33_v35_specialists",
        "classes": CLASSES,
        "text_format": "[CURRENT]\\n...\\n[HISTORY]\\n...\\n[META]\\n...\\n[WORKSPACE]\\n...",
        "specialists": {
            "read_list": {
                "artifact_dir": "read_list",
                "pair": [READ_ID, LIST_ID],
                "C": 1.0,
                "direction": "both",
                "qwen_margin_max": 0.75,
                "specialist_probability_min": 0.50,
                "source_probe_gain_qwen": 0.0007492688486053778,
                "source_probe_gain_v23_full": 0.000386160834759397,
            },
            "ask_plan": {
                "artifact_dir": "ask_plan",
                "pair": [ASK_ID, PLAN_ID],
                "C": 0.25,
                "direction": "ask_to_plan_only",
                "qwen_margin_max": 1.5,
                "specialist_probability_min": 0.55,
                "source_probe_gain_qwen": 0.001053601,
                "source_probe_gain_v23_full": 0.000830099,
            },
        },
        "combined_probe": {
            "qwen_baseline": 0.784299793,
            "qwen_combined": 0.786102664,
            "qwen_gain": 0.001802870,
            "v23_full_baseline": 0.785684405,
            "v23_full_combined": 0.786900665,
            "v23_full_gain": 0.001216260,
        },
    }

    (args.output_dir / "specialist_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
