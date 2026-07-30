import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "analysis"

NPZ_PATH = MODEL_DIR / "ensemble_validation_outputs.npz"

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]


def load_data():
    samples = []

    with (DATA_DIR / "train.jsonl").open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    with (DATA_DIR / "train_labels.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        labels = {
            row["id"]: row["action"]
            for row in csv.DictReader(file)
        }

    return samples, labels


def get_history_info(sample):
    history = sample.get("history", [])

    if not isinstance(history, list):
        history = []

    actions = []
    last_result = ""

    for item in history:
        if not isinstance(item, dict):
            continue

        name = item.get("name")

        if name:
            actions.append(str(name))
            last_result = str(item.get("result_summary", ""))

    return {
        "history_len": len(history),
        "action_count": len(actions),
        "last_action": actions[-1] if actions else "none",
        "second_last_action": actions[-2] if len(actions) >= 2 else "none",
        "action_sequence": " > ".join(actions[-4:]),
        "last_result": last_result,
    }


def get_meta_info(sample):
    meta = sample.get("session_meta", {})

    if not isinstance(meta, dict):
        meta = {}

    workspace = meta.get("workspace", {})

    if not isinstance(workspace, dict):
        workspace = {}

    open_files = workspace.get("open_files", [])

    if not isinstance(open_files, list):
        open_files = []

    language_mix = workspace.get("language_mix", {})

    if not isinstance(language_mix, dict):
        language_mix = {}

    top_language = "none"

    if language_mix:
        top_language = max(
            language_mix.items(),
            key=lambda item: item[1],
        )[0]

    return {
        "user_tier": meta.get("user_tier", "unknown"),
        "language_pref": meta.get("language_pref", "unknown"),
        "budget_tokens_remaining": meta.get(
            "budget_tokens_remaining",
            -1,
        ),
        "turn_index": meta.get("turn_index", -1),
        "elapsed_session_sec": meta.get(
            "elapsed_session_sec",
            -1,
        ),
        "git_dirty": workspace.get("git_dirty", "unknown"),
        "last_ci_status": workspace.get(
            "last_ci_status",
            "unknown",
        ),
        "workspace_loc": workspace.get("loc", -1),
        "open_file_count": len(open_files),
        "open_files": " | ".join(map(str, open_files)),
        "top_language": top_language,
    }


def prompt_flags(prompt):
    text = str(prompt or "")
    lower = text.lower()

    file_path_pattern = re.compile(
        r"[\w./\\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|css|html|"
        r"json|ya?ml|md|xml|toml|gradle|kt|swift|php|rb|sql)\b",
        re.IGNORECASE,
    )

    return {
        "prompt_length": len(text),
        "has_file_path": bool(file_path_pattern.search(text)),
        "has_find_word": bool(
            re.search(
                r"찾|검색|어디|locate|find|search|grep|사용처|호출처",
                lower,
            )
        ),
        "has_open_word": bool(
            re.search(
                r"열어|보여|읽어|내용|훑어|봐줘|open|read|show",
                lower,
            )
        ),
        "has_list_word": bool(
            re.search(
                r"목록|리스트|구조|디렉터리|폴더|파일들|list|directory|"
                r"folder|how many files",
                lower,
            )
        ),
        "has_test_word": bool(
            re.search(
                r"테스트|회귀|pytest|vitest|jest|unittest|test",
                lower,
            )
        ),
        "has_lint_word": bool(
            re.search(
                r"린트|타입|타입체크|문법|빌드|eslint|mypy|tsc|"
                r"typecheck|lint|compile",
                lower,
            )
        ),
        "has_web_word": bool(
            re.search(
                r"공식|문서|권장|최신|버전|검색해|찾아봐|인터넷|"
                r"documentation|official|latest|recommended",
                lower,
            )
        ),
        "has_plan_word": bool(
            re.search(
                r"계획|단계|순서|접근|정리해|plan|steps|approach|"
                r"break.*down",
                lower,
            )
        ),
    }


def normalize_prompt(prompt):
    text = str(prompt or "").lower().strip()
    text = re.sub(r"\b\d+\b", "<num>", text)
    text = re.sub(r"\s+", " ", text)
    return text


def make_rows(samples, labels, validation_indices, probabilities):
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)

    rows = []

    for local_index, sample_index in enumerate(validation_indices):
        sample = samples[int(sample_index)]
        sample_id = str(sample.get("id", ""))
        true_action = labels[sample_id]
        predicted_action = ALL_CLASSES[int(predictions[local_index])]

        row = {
            "sample_index": int(sample_index),
            "id": sample_id,
            "true_action": true_action,
            "pred_action": predicted_action,
            "correct": true_action == predicted_action,
            "confidence": float(confidences[local_index]),
            "current_prompt": sample.get("current_prompt", ""),
            "source": (
                "sim" if sample_id.startswith("sess_sim_")
                else "au" if sample_id.startswith("sess_au_")
                else "other"
            ),
        }

        row.update(get_history_info(sample))
        row.update(get_meta_info(sample))
        row.update(prompt_flags(row["current_prompt"]))

        true_id = ALL_CLASSES.index(true_action)
        row["true_probability"] = float(
            probabilities[local_index, true_id]
        )

        for class_index, class_name in enumerate(ALL_CLASSES):
            row[f"prob_{class_name}"] = float(
                probabilities[local_index, class_index]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_ambiguous_prompt_report(samples, labels):
    groups = defaultdict(Counter)
    examples = defaultdict(list)

    for sample in samples:
        prompt = normalize_prompt(
            sample.get("current_prompt", "")
        )
        action = labels[str(sample["id"])]

        groups[prompt][action] += 1

        if len(examples[prompt]) < 5:
            examples[prompt].append(str(sample["id"]))

    rows = []

    for prompt, counts in groups.items():
        if len(counts) <= 1:
            continue

        total = sum(counts.values())
        majority_count = max(counts.values())

        rows.append({
            "normalized_prompt": prompt,
            "total_count": total,
            "label_count": len(counts),
            "majority_ratio": majority_count / total,
            "label_distribution": json.dumps(
                counts,
                ensure_ascii=False,
            ),
            "example_ids": " | ".join(examples[prompt]),
        })

    return pd.DataFrame(rows).sort_values(
        ["total_count", "label_count"],
        ascending=False,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"검증 출력 파일이 없습니다: {NPZ_PATH}"
        )

    samples, labels = load_data()
    outputs = np.load(NPZ_PATH)

    validation_indices = outputs["validation_indices"]
    validation_labels = outputs["validation_labels"]

    if "ensemble_probabilities" in outputs:
        probabilities = outputs["ensemble_probabilities"]
        model_name = "ensemble"
    else:
        logits = outputs["qwen_logits"]
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        model_name = "qwen"

    frame = make_rows(
        samples,
        labels,
        validation_indices,
        probabilities,
    )

    frame.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    y_true = frame["true_action"]
    y_pred = frame["pred_action"]

    report = classification_report(
        y_true,
        y_pred,
        labels=ALL_CLASSES,
        target_names=ALL_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    report_frame = pd.DataFrame(report).T
    report_frame.to_csv(
        OUTPUT_DIR / "class_report.csv",
        encoding="utf-8-sig",
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=ALL_CLASSES,
    )

    matrix_frame = pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in ALL_CLASSES],
        columns=[f"pred_{name}" for name in ALL_CLASSES],
    )
    matrix_frame.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    confusion_rows = []

    for true_index, true_name in enumerate(ALL_CLASSES):
        for pred_index, pred_name in enumerate(ALL_CLASSES):
            if true_name == pred_name:
                continue

            count = int(matrix[true_index, pred_index])

            if count:
                confusion_rows.append({
                    "true_action": true_name,
                    "pred_action": pred_name,
                    "count": count,
                    "true_total": int(matrix[true_index].sum()),
                    "error_ratio_within_true": (
                        count / max(1, matrix[true_index].sum())
                    ),
                })

    confusion_pairs = pd.DataFrame(
        confusion_rows
    ).sort_values(
        ["count", "error_ratio_within_true"],
        ascending=False,
    )

    confusion_pairs.to_csv(
        OUTPUT_DIR / "top_confusions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    high_confidence_errors = frame[
        ~frame["correct"]
    ].sort_values(
        ["confidence", "true_probability"],
        ascending=[False, True],
    )

    high_confidence_errors.to_csv(
        OUTPUT_DIR / "high_confidence_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_columns = [
        "source",
        "last_action",
        "second_last_action",
        "language_pref",
        "user_tier",
        "last_ci_status",
        "top_language",
        "has_file_path",
        "has_find_word",
        "has_open_word",
        "has_list_word",
        "has_test_word",
        "has_lint_word",
        "has_web_word",
        "has_plan_word",
    ]

    segment_rows = []

    for column in group_columns:
        for value, group in frame.groupby(column, dropna=False):
            if len(group) < 20:
                continue

            segment_rows.append({
                "feature": column,
                "value": value,
                "samples": len(group),
                "accuracy": group["correct"].mean(),
                "mean_confidence": group["confidence"].mean(),
            })

    pd.DataFrame(segment_rows).sort_values(
        ["feature", "samples"],
        ascending=[True, False],
    ).to_csv(
        OUTPUT_DIR / "segment_accuracy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ambiguous = build_ambiguous_prompt_report(
        samples,
        labels,
    )
    ambiguous.to_csv(
        OUTPUT_DIR / "ambiguous_prompts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for true_action, pred_action in (
        confusion_pairs.head(15)[
            ["true_action", "pred_action"]
        ].itertuples(index=False, name=None)
    ):
        pair = frame[
            (frame["true_action"] == true_action)
            & (frame["pred_action"] == pred_action)
        ].sort_values(
            "confidence",
            ascending=False,
        )

        pair.to_csv(
            OUTPUT_DIR
            / f"errors_{true_action}_as_{pred_action}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(f"Model outputs: {model_name}")
    print(f"Validation samples: {len(frame)}")
    print()
    print("Top confusion pairs:")

    print(
        confusion_pairs.head(20).to_string(
            index=False,
        )
    )

    print()
    print("Saved reports:")
    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        print(" ", path)


if __name__ == "__main__":
    main()
