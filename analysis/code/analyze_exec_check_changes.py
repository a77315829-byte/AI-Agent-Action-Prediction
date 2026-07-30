#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract and analyze samples changed by the V12 + V24 1.5B exec/check gated blend.

Target classes:
- run_bash
- run_tests
- lint_or_typecheck

Example:
python .\analyze_exec_check_changes.py `
  --data .\data\train.jsonl `
  --labels .\data\train_labels.csv `
  --v12-logits .\model\qwen_distill_v12_eval\validation_logits_v12.npz `
  --v15-logits .\model\qwen_segment_v24_15b_r16_eval\validation_logits_v24_15b.npz `
  --v15-postprocess .\model\qwen_segment_v24_15b_r16_eval\postprocess.json `
  --output-dir .\model\exec_check_change_analysis `
  --v15-weight 0.60
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
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

LABEL2ID = {label: index for index, label in enumerate(CLASSES)}
ACTION_TO_FAMILY = np.asarray(
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 3],
    dtype=np.int64,
)

EXEC_CHECK_IDS = np.asarray(
    [
        LABEL2ID["run_bash"],
        LABEL2ID["run_tests"],
        LABEL2ID["lint_or_typecheck"],
    ],
    dtype=np.int64,
)

V12_TRAINING_CLASS_WEIGHTS = np.asarray(
    [
        0.8572690251352013,
        0.84274223991845,
        1.0366924510397455,
        0.9862908339428644,
        0.8179509647354836,
        1.3554499048325421,
        1.009072807992434,
        0.996587290637956,
        1.0232914644762283,
        1.2165721371457607,
        1.166404985555288,
        1.1687102789899324,
        1.407913957899642,
        0.9913123924717526,
    ],
    dtype=np.float64,
)

PATTERNS = {
    "has_test_word": r"\b(test|tests|testing|pytest|unittest|jest|vitest|mocha)\b|테스트",
    "has_run_test_phrase": r"\b(run|execute|rerun)\s+(the\s+)?(test|tests|test suite)\b|테스트.*실행|실행.*테스트",
    "has_pytest": r"\bpytest\b",
    "has_npm_test": r"\b(npm|pnpm|yarn)\s+(run\s+)?test\b",
    "has_unit_test": r"\b(unit|integration|e2e|end[- ]to[- ]end)\s+tests?\b",
    "has_lint": r"\b(lint|eslint|pylint|ruff|flake8|stylelint)\b|린트",
    "has_typecheck": r"\b(typecheck|type-check|type check|tsc|mypy|pyright|type error)\b|타입\s*체크|타입검사",
    "has_build": r"\b(build|compile|compilation|make|cmake)\b|빌드|컴파일",
    "has_install": r"\b(install|pip install|npm install|pnpm install|yarn install)\b|설치",
    "has_server_run": r"\b(start|launch|serve|server|dev server|npm run dev)\b|서버.*실행|실행.*서버",
    "has_shell_command": r"\b(command|terminal|shell|bash|powershell|cmd|execute)\b|명령어|터미널|쉘",
    "has_error_fix": r"\b(error|failed|failure|traceback|exception|fix)\b|에러|오류|실패",
    "has_check_verify": r"\b(check|verify|validate|validation)\b|확인|검증",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--v12-logits", type=Path, required=True)
    parser.add_argument("--v15-logits", type=Path, required=True)
    parser.add_argument("--v15-postprocess", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--v15-weight", type=float, default=0.60)
    parser.add_argument("--eval-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--v12-action-temperature", type=float, default=0.6)
    parser.add_argument("--v12-family-weight", type=float, default=0.15)
    parser.add_argument("--v12-prior-beta", type=float, default=0.25)

    parser.add_argument("--history-preview-chars", type=int, default=1200)
    parser.add_argument("--current-preview-chars", type=int, default=2000)
    return parser.parse_args()


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max(axis=1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-300)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labels(path: Path, samples: List[dict]) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as file:
        label_map = {
            str(row["id"]): str(row["action"])
            for row in csv.DictReader(file)
        }

    labels: List[int] = []
    for sample in samples:
        sample_id = str(sample["id"])
        if sample_id not in label_map:
            raise RuntimeError(f"Missing label: {sample_id}")
        action = label_map[sample_id]
        if action not in LABEL2ID:
            raise RuntimeError(f"Unknown action label: {action}")
        labels.append(LABEL2ID[action])
    return np.asarray(labels, dtype=np.int64)


def resolve_validation_indices(
    samples: List[dict],
    labels: np.ndarray,
    eval_fold: int,
    seed: int,
) -> np.ndarray:
    groups = np.asarray(
        [str(sample["id"]).rsplit("-step_", 1)[0] for sample in samples]
    )
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    for fold, (_, validation_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups)
    ):
        if fold == eval_fold:
            return np.asarray(validation_indices, dtype=np.int64)

    raise RuntimeError(f"Invalid eval_fold={eval_fold}")


def compute_probability(
    payload: Dict[str, np.ndarray],
    action_temperature: float,
    family_weight: float,
    prior_beta: float,
    training_class_weights: np.ndarray | None,
) -> np.ndarray:
    final_logits = (
        payload["action_logits"].astype(np.float64)
        / float(action_temperature)
    )

    if family_weight != 0:
        final_logits += (
            float(family_weight)
            * payload["family_logits"].astype(np.float64)[:, ACTION_TO_FAMILY]
        )

    if prior_beta != 0:
        if training_class_weights is None:
            raise RuntimeError(
                "training_class_weights required when prior_beta != 0"
            )
        final_logits -= (
            float(prior_beta)
            * np.log(np.maximum(training_class_weights, 1e-12))[None, :]
        )

    return stable_softmax(final_logits)


def load_v15_postprocess(path: Path) -> dict:
    postprocess = json.loads(path.read_text(encoding="utf-8"))
    return {
        "action_temperature": float(
            postprocess.get("action_temperature", 1.0)
        ),
        "family_weight": float(postprocess.get("family_weight", 0.0)),
        "prior_beta": float(postprocess.get("prior_beta", 0.0)),
        "training_class_weights": postprocess.get("training_class_weights"),
    }


def log_probability_blend(
    base_probability: np.ndarray,
    specialist_probability: np.ndarray,
    specialist_weight: float,
) -> np.ndarray:
    weight = float(specialist_weight)
    blended_log_probability = (
        (1.0 - weight)
        * np.log(np.maximum(base_probability, 1e-12))
        + weight
        * np.log(np.maximum(specialist_probability, 1e-12))
    )
    return stable_softmax(blended_log_probability)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(len(CLASSES))),
            average="macro",
            zero_division=0,
        )
    )


def stringify(value) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            else:
                chunks.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(chunks)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


def get_current_prompt(sample: dict) -> str:
    for key in ("current_prompt", "prompt", "instruction", "query"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def get_history(sample: dict) -> str:
    for key in ("history", "messages", "conversation"):
        if key in sample:
            return stringify(sample.get(key))
    return ""


def trim_tail(text: str, max_chars: int) -> str:
    text = text.replace("\r\n", "\n")
    if len(text) <= max_chars:
        return text
    return "...[truncated]...\n" + text[-max_chars:]


def probability_margin(probability_row: np.ndarray) -> float:
    top_two = np.partition(probability_row, -2)[-2:]
    return float(top_two.max() - top_two.min())


def exec_probability_margin(probability_row: np.ndarray) -> float:
    subset = probability_row[EXEC_CHECK_IDS]
    top_two = np.partition(subset, -2)[-2:]
    return float(top_two.max() - top_two.min())


def extract_pattern_flags(text: str) -> Dict[str, int]:
    normalized = text.lower()
    return {
        name: int(bool(re.search(pattern, normalized, flags=re.I)))
        for name, pattern in PATTERNS.items()
    }


def top_two_names(probability_row: np.ndarray) -> tuple[str, str]:
    order = np.argsort(probability_row)[::-1]
    return CLASSES[int(order[0])], CLASSES[int(order[1])]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(args.data)
    labels_all = load_labels(args.labels, samples)
    validation_indices = resolve_validation_indices(
        samples,
        labels_all,
        args.eval_fold,
        args.seed,
    )

    v12 = load_npz(args.v12_logits)
    v15 = load_npz(args.v15_logits)

    required_keys = {"action_logits", "family_logits", "labels"}
    for name, payload in (("V12", v12), ("V15", v15)):
        missing = required_keys - set(payload)
        if missing:
            raise RuntimeError(f"{name} missing keys: {sorted(missing)}")

    if not np.array_equal(v12["labels"], v15["labels"]):
        raise RuntimeError("V12 and V15 labels are not identical.")

    if "validation_indices" in v15:
        if not np.array_equal(
            v15["validation_indices"].astype(np.int64),
            validation_indices,
        ):
            raise RuntimeError(
                "V15 validation_indices do not match reconstructed fold."
            )

    y = labels_all[validation_indices]
    if not np.array_equal(y, v12["labels"].astype(np.int64)):
        raise RuntimeError(
            "Logit labels do not match reconstructed validation labels."
        )

    validation_samples = [samples[int(index)] for index in validation_indices]

    v12_probability = compute_probability(
        v12,
        action_temperature=args.v12_action_temperature,
        family_weight=args.v12_family_weight,
        prior_beta=args.v12_prior_beta,
        training_class_weights=V12_TRAINING_CLASS_WEIGHTS,
    )

    v15_postprocess = load_v15_postprocess(args.v15_postprocess)
    v15_class_weights = v15_postprocess["training_class_weights"]
    if v15_class_weights is not None:
        v15_class_weights = np.asarray(v15_class_weights, dtype=np.float64)

    v15_probability = compute_probability(
        v15,
        action_temperature=v15_postprocess["action_temperature"],
        family_weight=v15_postprocess["family_weight"],
        prior_beta=v15_postprocess["prior_beta"],
        training_class_weights=v15_class_weights,
    )

    v12_prediction = v12_probability.argmax(axis=1)
    v15_prediction = v15_probability.argmax(axis=1)

    blended_probability = log_probability_blend(
        v12_probability,
        v15_probability,
        args.v15_weight,
    )
    blended_prediction = blended_probability.argmax(axis=1)

    eligible = (
        np.isin(v12_prediction, EXEC_CHECK_IDS)
        & np.isin(v15_prediction, EXEC_CHECK_IDS)
    )

    gated_prediction = v12_prediction.copy()
    gated_prediction[eligible] = blended_prediction[eligible]

    changed = gated_prediction != v12_prediction
    changed_indices = np.flatnonzero(changed)

    rows = []
    for local_index in changed_indices:
        sample = validation_samples[int(local_index)]
        current_prompt = get_current_prompt(sample)
        history = get_history(sample)
        combined_text = f"{history}\n{current_prompt}"

        true_id = int(y[local_index])
        v12_id = int(v12_prediction[local_index])
        v15_id = int(v15_prediction[local_index])
        gated_id = int(gated_prediction[local_index])

        v12_correct = v12_id == true_id
        gated_correct = gated_id == true_id

        if not v12_correct and gated_correct:
            outcome = "gain"
        elif v12_correct and not gated_correct:
            outcome = "loss"
        elif not v12_correct and not gated_correct:
            outcome = "changed_wrong_to_wrong"
        else:
            outcome = "changed_correct_to_correct"

        v12_top1, v12_top2 = top_two_names(v12_probability[local_index])
        v15_top1, v15_top2 = top_two_names(v15_probability[local_index])

        row = {
            "local_val_index": int(local_index),
            "global_row_index": int(validation_indices[local_index]),
            "id": str(sample.get("id", "")),
            "true_id": true_id,
            "true_action": CLASSES[true_id],
            "v12_pred": CLASSES[v12_id],
            "v15_pred": CLASSES[v15_id],
            "gated_pred": CLASSES[gated_id],
            "outcome": outcome,
            "v12_correct": int(v12_correct),
            "v15_correct": int(v15_id == true_id),
            "gated_correct": int(gated_correct),
            "v12_top1": v12_top1,
            "v12_top2": v12_top2,
            "v15_top1": v15_top1,
            "v15_top2": v15_top2,
            "v12_global_margin": probability_margin(
                v12_probability[local_index]
            ),
            "v15_global_margin": probability_margin(
                v15_probability[local_index]
            ),
            "v12_exec_margin": exec_probability_margin(
                v12_probability[local_index]
            ),
            "v15_exec_margin": exec_probability_margin(
                v15_probability[local_index]
            ),
            "v12_run_bash_prob": float(
                v12_probability[local_index, LABEL2ID["run_bash"]]
            ),
            "v12_run_tests_prob": float(
                v12_probability[local_index, LABEL2ID["run_tests"]]
            ),
            "v12_lint_prob": float(
                v12_probability[
                    local_index,
                    LABEL2ID["lint_or_typecheck"],
                ]
            ),
            "v15_run_bash_prob": float(
                v15_probability[local_index, LABEL2ID["run_bash"]]
            ),
            "v15_run_tests_prob": float(
                v15_probability[local_index, LABEL2ID["run_tests"]]
            ),
            "v15_lint_prob": float(
                v15_probability[
                    local_index,
                    LABEL2ID["lint_or_typecheck"],
                ]
            ),
            "blend_run_bash_prob": float(
                blended_probability[local_index, LABEL2ID["run_bash"]]
            ),
            "blend_run_tests_prob": float(
                blended_probability[local_index, LABEL2ID["run_tests"]]
            ),
            "blend_lint_prob": float(
                blended_probability[
                    local_index,
                    LABEL2ID["lint_or_typecheck"],
                ]
            ),
            **extract_pattern_flags(combined_text),
            "current_prompt": trim_tail(
                current_prompt,
                args.current_preview_chars,
            ),
            "history_tail": trim_tail(
                history,
                args.history_preview_chars,
            ),
        }
        rows.append(row)

    changed_df = pd.DataFrame(rows)

    if len(changed_df) == 0:
        raise RuntimeError("No predictions changed under the current gate.")

    outcome_summary = (
        changed_df.groupby("outcome", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    transition_summary = (
        changed_df.groupby(
            ["v12_pred", "gated_pred", "true_action", "outcome"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    pattern_columns = list(PATTERNS)
    pattern_summary_rows = []
    for pattern_name in pattern_columns:
        for outcome, part in changed_df.groupby("outcome"):
            pattern_summary_rows.append(
                {
                    "pattern": pattern_name,
                    "outcome": outcome,
                    "matched": int(part[pattern_name].sum()),
                    "rows": int(len(part)),
                    "match_rate": float(part[pattern_name].mean()),
                }
            )

    pattern_summary = pd.DataFrame(pattern_summary_rows)

    score_v12 = macro_f1(y, v12_prediction)
    score_gated = macro_f1(y, gated_prediction)

    gains = int((changed_df["outcome"] == "gain").sum())
    losses = int((changed_df["outcome"] == "loss").sum())
    wrong_to_wrong = int(
        (changed_df["outcome"] == "changed_wrong_to_wrong").sum()
    )

    summary = {
        "rows": int(len(y)),
        "changed_rows": int(len(changed_df)),
        "eligible_rows": int(eligible.sum()),
        "v15_weight": float(args.v15_weight),
        "v12_macro_f1": score_v12,
        "gated_macro_f1": score_gated,
        "gain_vs_v12": score_gated - score_v12,
        "gain_count": gains,
        "loss_count": losses,
        "changed_wrong_to_wrong_count": wrong_to_wrong,
        "net_correct_changes": gains - losses,
        "v15_postprocess": v15_postprocess,
    }

    changed_df.to_csv(
        args.output_dir / "exec_check_changed_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    changed_df[changed_df["outcome"] == "gain"].to_csv(
        args.output_dir / "exec_check_gains.csv",
        index=False,
        encoding="utf-8-sig",
    )

    changed_df[changed_df["outcome"] == "loss"].to_csv(
        args.output_dir / "exec_check_losses.csv",
        index=False,
        encoding="utf-8-sig",
    )

    outcome_summary.to_csv(
        args.output_dir / "outcome_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    transition_summary.to_csv(
        args.output_dir / "transition_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pattern_summary.to_csv(
        args.output_dir / "pattern_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== Exec/check gated change analysis ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n=== Outcome summary ===")
    print(outcome_summary.to_string(index=False))

    print("\n=== Top transitions ===")
    print(transition_summary.head(30).to_string(index=False))

    print("\n=== Top gain patterns ===")
    gain_patterns = (
        pattern_summary[pattern_summary["outcome"] == "gain"]
        .sort_values(["match_rate", "matched"], ascending=[False, False])
    )
    print(gain_patterns.head(20).to_string(index=False))

    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
