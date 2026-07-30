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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm

from feature_utils_qwen_v4 import ACTION_TO_FAMILY, ALL_CLASSES, extract_state


SEED = 42
NUM_CLASSES = len(ALL_CLASSES)
LABEL2ID = {label: i for i, label in enumerate(ALL_CLASSES)}
FAMILY_INDEX = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)

DOMAIN_NAMES = ["explore", "modify", "execute", "decision"]
DOMAIN_TO_ACTIONS = [
    ["read_file", "grep_search", "list_directory", "glob_pattern"],
    ["edit_file", "write_file", "apply_patch"],
    ["run_bash", "run_tests", "lint_or_typecheck"],
    ["ask_user", "plan_task", "web_search", "respond_only"],
]
DOMAIN_ACTION_IDS = [[LABEL2ID[x] for x in xs] for xs in DOMAIN_TO_ACTIONS]
ACTION_TO_DOMAIN = np.zeros(NUM_CLASSES, dtype=np.int64)
for domain_id, action_ids in enumerate(DOMAIN_ACTION_IDS):
    for action_id in action_ids:
        ACTION_TO_DOMAIN[action_id] = domain_id


URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|(?:\.{0,2}[/\\]))"
    r"(?:[^\s<>:\"|?*]+[/\\])*[^\s<>:\"|?*]*"
)
FILE_RE = re.compile(
    r"\b[\w.@+~-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cpp|cc|c|h|hpp|"
    r"cs|css|html|json|ya?ml|md|xml|toml|ini|env|gradle|kt|swift|"
    r"php|rb|sql|sh|ps1|bat|vue|svelte|txt|csv|lock)\b",
    re.IGNORECASE,
)
GLOB_RE = re.compile(r"(?:\*\*/|\*\.[A-Za-z0-9]+|[?*][\w.-]*|\[[^\]]+\])")
NUMBER_RE = re.compile(r"(?<![\w])[-+]?\d+(?:\.\d+)?(?![\w])")
HEX_RE = re.compile(r"\b[0-9a-f]{7,}\b", re.IGNORECASE)
QUOTED_RE = re.compile(r"`[^`]{1,120}`|'[^']{1,120}'|\"[^\"]{1,120}\"")
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./@+-]*|[가-힣]{2,}")
STEP_RE = re.compile(r"-step_(\d+)")

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "you",
    "are", "was", "were", "have", "has", "had", "not", "but", "can", "will",
    "should", "please", "file", "files", "code", "project", "need", "use",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V11 Template-Sequence Retrieval Policy. This is a retrieval/rule "
            "policy that tries to recover the dataset's state-transition "
            "templates rather than training another classifier."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--oof-logits", type=Path, default=Path("model/qwen_v4_oof/oof_logits_all_70000.npz"))
    parser.add_argument("--postprocess", type=Path, default=Path("model/qwen_segment_v4/postprocess.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/template_sequence_retrieval_v11"))
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--smoothing", type=float, default=0.15)
    parser.add_argument("--max-table-size-per-spec", type=int, default=0, help="0 means keep all keys.")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_samples_and_labels(data_path: Path, labels_path: Path) -> Tuple[List[dict], np.ndarray]:
    with labels_path.open(encoding="utf-8", newline="") as file:
        label_map = {str(row["id"]): LABEL2ID[str(row["action"])] for row in csv.DictReader(file)}

    samples: List[dict] = []
    labels: List[int] = []
    with data_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            sample_id = str(sample.get("id", ""))
            if sample_id not in label_map:
                raise KeyError(f"Missing label for id={sample_id}, line={line_number}")
            samples.append(sample)
            labels.append(label_map[sample_id])

    return samples, np.asarray(labels, dtype=np.int64)


def compact_text(value: Any, limit: int = 2000) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except TypeError:
            text = str(value)
    return WHITESPACE_RE.sub(" ", text).strip()[:limit]


def normalize_text(value: Any, limit: int = 1600) -> str:
    text = compact_text(value, limit).casefold()
    text = URL_RE.sub(" <URL> ", text)
    text = PATH_RE.sub(" <PATH> ", text)
    text = FILE_RE.sub(" <FILE> ", text)
    text = GLOB_RE.sub(" <GLOB> ", text)
    text = HEX_RE.sub(" <HEX> ", text)
    text = NUMBER_RE.sub(" <NUM> ", text)
    text = QUOTED_RE.sub(" <QUOTE> ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def stable_tokens(text: str, max_tokens: int = 24) -> str:
    normalized = normalize_text(text, 1600)
    tokens = []
    for token in WORD_RE.findall(normalized):
        token = token.casefold()
        if len(token) <= 1 or token in STOPWORDS:
            continue
        tokens.append(token)
    counter = Counter(tokens)
    ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:max_tokens]
    return "_".join(token for token, _ in ranked)


def short_hash_text(text: str, size: int = 20) -> str:
    tokens = stable_tokens(text, max_tokens=size)
    if not tokens:
        return "empty"
    return tokens


def step_number(sample: Mapping[str, Any]) -> int:
    match = STEP_RE.search(str(sample.get("id", "")))
    if not match:
        return 0
    return int(match.group(1))


def source_name(sample: Mapping[str, Any]) -> str:
    sample_id = str(sample.get("id", ""))
    if sample_id.startswith("sess_sim_"):
        return "sim"
    if sample_id.startswith("sess_au_"):
        return "au"
    return "other"


def lexical_signature(text: Any) -> str:
    lower = compact_text(text, 1500).casefold()
    flags: List[str] = []

    if PATH_RE.search(lower):
        flags.append("path")
    if FILE_RE.search(lower):
        flags.append("file")
    if GLOB_RE.search(lower):
        flags.append("glob")
    if "traceback" in lower or "exception" in lower or "error" in lower:
        flags.append("error")
    if "fail" in lower or "failed" in lower or "failure" in lower:
        flags.append("fail")
    if "passed" in lower or "success" in lower or re.search(r"\bok\b", lower):
        flags.append("success")
    if "test" in lower or "pytest" in lower or "unittest" in lower:
        flags.append("test")
    if "lint" in lower or "typecheck" in lower or "mypy" in lower or "tsc" in lower:
        flags.append("lint")
    if "grep" in lower or "search" in lower or "find" in lower:
        flags.append("search")
    if "read" in lower or "open" in lower or "show" in lower or "view" in lower:
        flags.append("read")
    if "directory" in lower or "folder" in lower or "list" in lower or "tree" in lower:
        flags.append("list")
    if "patch" in lower or "diff" in lower or "edit" in lower or "modify" in lower:
        flags.append("modify")
    if "bash" in lower or "command" in lower or "terminal" in lower or "run" in lower:
        flags.append("run")
    if "web" in lower or "browser" in lower or "internet" in lower:
        flags.append("web")
    if "?" in lower or "clarify" in lower or "ask" in lower or "unclear" in lower:
        flags.append("question")
    if "not found" in lower or "no such file" in lower or "cannot find" in lower:
        flags.append("notfound")
    if "permission" in lower or "denied" in lower:
        flags.append("denied")
    if "empty" in lower or "no results" in lower or "nothing" in lower:
        flags.append("empty")
    if "json" in lower:
        flags.append("json")
    if "python" in lower or ".py" in lower:
        flags.append("py")
    if "typescript" in lower or ".ts" in lower or ".tsx" in lower:
        flags.append("ts")
    if not flags:
        flags.append("none")

    return "+".join(sorted(set(flags)))


def result_signature(action_items: Sequence[Mapping[str, Any]]) -> str:
    if not action_items:
        return "no_tool"

    last = action_items[-1]
    name = str(last.get("name", "unknown"))
    result = compact_text(last.get("result", ""), 1600).casefold()
    args = compact_text(last.get("args", ""), 1000).casefold()

    sig = [f"tool={name}", f"res={lexical_signature(result)}"]
    if args:
        sig.append(f"args={lexical_signature(args)}")
    if len(result) < 20:
        sig.append("short")
    elif len(result) > 1000:
        sig.append("long")
    else:
        sig.append("mid")

    return "|".join(sig)


def prompt_intent_signature(current: str) -> str:
    normalized = normalize_text(current, 1200)
    return "|".join([
        f"lex={lexical_signature(current)}",
        f"tok={short_hash_text(normalized, 16)}",
    ])


def action_sequence(previous_actions: Sequence[str], n: int) -> str:
    selected = list(previous_actions[-n:])
    if not selected:
        return "none"
    return ">".join(selected)


def domain_sequence(previous_actions: Sequence[str], n: int) -> str:
    values = []
    for action in previous_actions[-n:]:
        if action in LABEL2ID:
            values.append(DOMAIN_NAMES[int(ACTION_TO_DOMAIN[LABEL2ID[action]])])
        else:
            values.append("unknown")
    if not values:
        return "none"
    return ">".join(values)


def final_logits(
    action_logits: np.ndarray,
    family_logits: np.ndarray,
    class_weights: np.ndarray,
    postprocess: Mapping[str, Any],
) -> np.ndarray:
    return (
        action_logits.astype(np.float64) / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"]) * family_logits.astype(np.float64)[:, FAMILY_INDEX]
        - float(postprocess["prior_beta"]) * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )


def macro_f1_fast(labels: np.ndarray, predictions: np.ndarray) -> float:
    confusion = np.bincount(
        labels * NUM_CLASSES + predictions,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    tp = np.diag(confusion).astype(np.float64)
    denominator = (confusion.sum(axis=0) + confusion.sum(axis=1)).astype(np.float64)
    class_f1 = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=denominator > 0,
    )
    return float(class_f1.mean())


def class_f1_values(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return f1_score(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
        average=None,
        zero_division=0,
    )


def build_signature(sample: Mapping[str, Any], qwen_pred: int | None = None, qwen_top2: int | None = None) -> Dict[str, str]:
    state = extract_state(dict(sample))
    current = compact_text(state.get("current_prompt", ""), 1600)
    current_norm = normalize_text(current, 1600)
    current_short = short_hash_text(current_norm, 18)
    current_tiny = short_hash_text(current_norm, 10)
    intent = prompt_intent_signature(current)

    previous_actions = [
        action for action in state.get("previous_actions", [])[-8:]
        if isinstance(action, str)
    ]
    action_items = state.get("action_items", [])[-4:]
    last_tool = str(action_items[-1].get("name", "none")) if action_items else "none"
    prev_tool = str(action_items[-2].get("name", "none")) if len(action_items) >= 2 else "none"

    result_sig = result_signature(action_items)
    source = source_name(sample)
    language = str(state.get("top_language", "unknown"))
    step = step_number(sample)
    step_bucket = (
        "s0" if step == 0 else
        "s1_2" if step <= 2 else
        "s3_5" if step <= 5 else
        "s6_10" if step <= 10 else
        "s11p"
    )

    last1 = action_sequence(previous_actions, 1)
    last2 = action_sequence(previous_actions, 2)
    last3 = action_sequence(previous_actions, 3)
    last4 = action_sequence(previous_actions, 4)
    dom2 = domain_sequence(previous_actions, 2)
    dom3 = domain_sequence(previous_actions, 3)
    dom4 = domain_sequence(previous_actions, 4)

    q1 = ALL_CLASSES[int(qwen_pred)] if qwen_pred is not None else "none"
    q2 = ALL_CLASSES[int(qwen_top2)] if qwen_top2 is not None else "none"
    qdom = DOMAIN_NAMES[int(ACTION_TO_DOMAIN[int(qwen_pred)])] if qwen_pred is not None else "none"

    return {
        "source": source,
        "language": language,
        "step_bucket": step_bucket,
        "current_norm": current_norm,
        "current_short": current_short,
        "current_tiny": current_tiny,
        "intent": intent,
        "lex": lexical_signature(current),
        "last_tool": last_tool,
        "prev_tool": prev_tool,
        "result_sig": result_sig,
        "last1": last1,
        "last2": last2,
        "last3": last3,
        "last4": last4,
        "dom2": dom2,
        "dom3": dom3,
        "dom4": dom4,
        "q1": q1,
        "q2": q2,
        "qdom": qdom,
    }


def build_keys(sig: Mapping[str, str]) -> Dict[str, str]:
    s = sig
    return {
        # Strict template-sequence keys.
        "src_cur_l4_tool_res": f"{s['source']}||{s['current_norm']}||{s['last4']}||{s['last_tool']}||{s['result_sig']}",
        "src_cur_l3_tool_res": f"{s['source']}||{s['current_norm']}||{s['last3']}||{s['last_tool']}||{s['result_sig']}",
        "src_cur_l3_tool": f"{s['source']}||{s['current_norm']}||{s['last3']}||{s['last_tool']}",
        "src_cur_l3": f"{s['source']}||{s['current_norm']}||{s['last3']}",
        "src_cur_l2_tool": f"{s['source']}||{s['current_norm']}||{s['last2']}||{s['last_tool']}",
        "src_cur_l2": f"{s['source']}||{s['current_norm']}||{s['last2']}",
        "cur_l3_tool_res": f"{s['current_norm']}||{s['last3']}||{s['last_tool']}||{s['result_sig']}",
        "cur_l3": f"{s['current_norm']}||{s['last3']}",
        "cur_l2": f"{s['current_norm']}||{s['last2']}",
        "cur_l1": f"{s['current_norm']}||{s['last1']}",
        "src_cur": f"{s['source']}||{s['current_norm']}",
        "cur": f"{s['current_norm']}",

        # Soft template keys based on stable keywords.
        "src_short_l3_tool_res": f"{s['source']}||{s['current_short']}||{s['last3']}||{s['last_tool']}||{s['result_sig']}",
        "src_short_l3_tool": f"{s['source']}||{s['current_short']}||{s['last3']}||{s['last_tool']}",
        "short_l3_tool": f"{s['current_short']}||{s['last3']}||{s['last_tool']}",
        "short_l2_tool_res": f"{s['current_short']}||{s['last2']}||{s['last_tool']}||{s['result_sig']}",
        "short_l2_tool": f"{s['current_short']}||{s['last2']}||{s['last_tool']}",
        "short_l2": f"{s['current_short']}||{s['last2']}",
        "tiny_l3_tool": f"{s['current_tiny']}||{s['last3']}||{s['last_tool']}",
        "tiny_l2_tool": f"{s['current_tiny']}||{s['last2']}||{s['last_tool']}",

        # Intent-state keys. These are less exact but much higher coverage.
        "src_intent_l3_tool_res": f"{s['source']}||{s['intent']}||{s['last3']}||{s['last_tool']}||{s['result_sig']}",
        "src_intent_l3_tool": f"{s['source']}||{s['intent']}||{s['last3']}||{s['last_tool']}",
        "src_intent_l2_tool": f"{s['source']}||{s['intent']}||{s['last2']}||{s['last_tool']}",
        "intent_l3_tool_res": f"{s['intent']}||{s['last3']}||{s['last_tool']}||{s['result_sig']}",
        "intent_l3_tool": f"{s['intent']}||{s['last3']}||{s['last_tool']}",
        "intent_l2_tool": f"{s['intent']}||{s['last2']}||{s['last_tool']}",
        "intent_l1_tool": f"{s['intent']}||{s['last1']}||{s['last_tool']}",
        "intent_dom3_tool_res": f"{s['intent']}||{s['dom3']}||{s['last_tool']}||{s['result_sig']}",
        "intent_dom3_tool": f"{s['intent']}||{s['dom3']}||{s['last_tool']}",
        "intent_dom2_tool": f"{s['intent']}||{s['dom2']}||{s['last_tool']}",

        # Tool-result transition keys.
        "src_l4_tool_res": f"{s['source']}||{s['last4']}||{s['last_tool']}||{s['result_sig']}",
        "src_l3_tool_res": f"{s['source']}||{s['last3']}||{s['last_tool']}||{s['result_sig']}",
        "src_l3_tool_lex": f"{s['source']}||{s['last3']}||{s['last_tool']}||{s['lex']}",
        "l3_tool_res": f"{s['last3']}||{s['last_tool']}||{s['result_sig']}",
        "l2_tool_res": f"{s['last2']}||{s['last_tool']}||{s['result_sig']}",
        "l1_tool_res": f"{s['last1']}||{s['last_tool']}||{s['result_sig']}",
        "dom4_tool_res": f"{s['dom4']}||{s['last_tool']}||{s['result_sig']}",
        "dom3_tool_res": f"{s['dom3']}||{s['last_tool']}||{s['result_sig']}",
        "dom2_tool_res": f"{s['dom2']}||{s['last_tool']}||{s['result_sig']}",

        # Qwen-conditioned retrieval keys.
        "q1_cur_l2": f"{s['q1']}||{s['current_norm']}||{s['last2']}",
        "q1_short_l3_tool": f"{s['q1']}||{s['current_short']}||{s['last3']}||{s['last_tool']}",
        "q1_intent_l3_tool": f"{s['q1']}||{s['intent']}||{s['last3']}||{s['last_tool']}",
        "qdom_intent_l3_tool": f"{s['qdom']}||{s['intent']}||{s['last3']}||{s['last_tool']}",
        "q1_l3_tool_res": f"{s['q1']}||{s['last3']}||{s['last_tool']}||{s['result_sig']}",
        "q1_q2_intent_l2": f"{s['q1']}||{s['q2']}||{s['intent']}||{s['last2']}",

        # Coarse buckets, used with lower weights.
        "lex_l3_tool": f"{s['lex']}||{s['last3']}||{s['last_tool']}",
        "lex_l2_tool": f"{s['lex']}||{s['last2']}||{s['last_tool']}",
        "source_lex_dom3_tool": f"{s['source']}||{s['lex']}||{s['dom3']}||{s['last_tool']}",
        "language_lex_l2_tool": f"{s['language']}||{s['lex']}||{s['last2']}||{s['last_tool']}",
        "step_lex_l2_tool": f"{s['step_bucket']}||{s['lex']}||{s['last2']}||{s['last_tool']}",
    }


def spec_weights() -> Dict[str, float]:
    return {
        # High-precision exact keys.
        "src_cur_l4_tool_res": 7.0,
        "src_cur_l3_tool_res": 6.5,
        "src_cur_l3_tool": 6.0,
        "src_cur_l3": 5.7,
        "src_cur_l2_tool": 5.5,
        "src_cur_l2": 5.2,
        "cur_l3_tool_res": 5.0,
        "cur_l3": 4.7,
        "cur_l2": 4.4,
        "cur_l1": 3.8,
        "src_cur": 3.6,
        "cur": 3.0,

        # Soft prompt keys.
        "src_short_l3_tool_res": 3.6,
        "src_short_l3_tool": 3.2,
        "short_l3_tool": 2.9,
        "short_l2_tool_res": 2.8,
        "short_l2_tool": 2.5,
        "short_l2": 2.2,
        "tiny_l3_tool": 1.8,
        "tiny_l2_tool": 1.5,

        # Intent-state keys.
        "src_intent_l3_tool_res": 3.2,
        "src_intent_l3_tool": 2.9,
        "src_intent_l2_tool": 2.6,
        "intent_l3_tool_res": 2.5,
        "intent_l3_tool": 2.2,
        "intent_l2_tool": 1.9,
        "intent_l1_tool": 1.5,
        "intent_dom3_tool_res": 2.0,
        "intent_dom3_tool": 1.7,
        "intent_dom2_tool": 1.4,

        # Tool transition keys.
        "src_l4_tool_res": 2.6,
        "src_l3_tool_res": 2.4,
        "src_l3_tool_lex": 2.1,
        "l3_tool_res": 2.0,
        "l2_tool_res": 1.8,
        "l1_tool_res": 1.5,
        "dom4_tool_res": 1.4,
        "dom3_tool_res": 1.2,
        "dom2_tool_res": 1.0,

        # Qwen-conditioned keys.
        "q1_cur_l2": 3.0,
        "q1_short_l3_tool": 2.6,
        "q1_intent_l3_tool": 2.3,
        "qdom_intent_l3_tool": 1.9,
        "q1_l3_tool_res": 2.1,
        "q1_q2_intent_l2": 2.0,

        # Coarse buckets.
        "lex_l3_tool": 1.0,
        "lex_l2_tool": 0.8,
        "source_lex_dom3_tool": 0.9,
        "language_lex_l2_tool": 0.7,
        "step_lex_l2_tool": 0.6,
    }


def make_signatures(samples: Sequence[dict], qwen_probs: np.ndarray) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    ranking = np.argsort(-qwen_probs, axis=1)
    signatures: List[Dict[str, str]] = []
    keys_list: List[Dict[str, str]] = []

    for index, sample in enumerate(tqdm(samples, desc="Build V11 signatures")):
        sig = build_signature(
            sample,
            qwen_pred=int(ranking[index, 0]),
            qwen_top2=int(ranking[index, 1]),
        )
        signatures.append(sig)
        keys_list.append(build_keys(sig))

    return signatures, keys_list


def table_from_indices(
    keys_list: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    indices: np.ndarray,
    specs: Sequence[str],
    min_count: int,
    max_table_size_per_spec: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    tables: Dict[str, Dict[str, np.ndarray]] = {}

    for spec in specs:
        counter: Dict[str, Counter] = defaultdict(Counter)
        for index in indices:
            key = keys_list[int(index)][spec]
            counter[key][int(labels[int(index)])] += 1

        items = []
        for key, counts in counter.items():
            total = sum(counts.values())
            if total < min_count:
                continue
            vector = np.zeros(NUM_CLASSES, dtype=np.float32)
            for label, count in counts.items():
                vector[int(label)] = float(count)
            purity = float(vector.max() / max(1.0, vector.sum()))
            items.append((key, vector, total, purity))

        if max_table_size_per_spec > 0 and len(items) > max_table_size_per_spec:
            items.sort(key=lambda row: (row[3], row[2]), reverse=True)
            items = items[:max_table_size_per_spec]

        tables[spec] = {key: vector for key, vector, _, _ in items}

    return tables


def predict_retrieval_with_tables(
    keys_list: Sequence[Mapping[str, str]],
    tables: Mapping[str, Mapping[str, np.ndarray]],
    specs: Sequence[str],
    weights: Mapping[str, float],
    smoothing: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(keys_list)
    probabilities = np.full((n, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float32)
    confidences = np.zeros(n, dtype=np.float32)
    hits = np.zeros(n, dtype=np.int16)
    masses = np.zeros(n, dtype=np.float32)
    best_purity = np.zeros(n, dtype=np.float32)

    for i, key_map in enumerate(tqdm(keys_list, desc="V11 retrieval vote", leave=False)):
        vote = np.full(NUM_CLASSES, smoothing, dtype=np.float64)
        total_weight = 0.0
        max_purity = 0.0
        hit_count = 0

        for spec in specs:
            vector = tables[spec].get(key_map[spec])
            if vector is None:
                continue

            total = float(vector.sum())
            if total <= 0:
                continue

            dist = (vector.astype(np.float64) + smoothing) / (total + smoothing * NUM_CLASSES)
            purity = float(vector.max() / total)
            spec_weight = float(weights[spec])
            count_weight = min(2.2, math.log1p(total) / math.log(8.0))
            purity_weight = 0.35 + purity
            weight = spec_weight * count_weight * purity_weight

            vote += weight * dist
            total_weight += weight
            max_purity = max(max_purity, purity)
            hit_count += 1

        if total_weight > 0:
            vote = vote / vote.sum()
            probabilities[i] = vote.astype(np.float32)
            confidences[i] = float(vote.max())
            hits[i] = hit_count
            masses[i] = min(1.0, total_weight / 25.0)
            best_purity[i] = max_purity

    predictions = probabilities.argmax(axis=1).astype(np.int64)
    return probabilities, predictions, confidences, hits, best_purity


def build_oof_retrieval(
    keys_list: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    fold_ids: np.ndarray,
    min_count: int,
    max_table_size_per_spec: int,
    smoothing: float,
) -> Dict[str, Any]:
    weights = spec_weights()
    specs = list(weights.keys())
    probabilities = np.full((len(labels), NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float32)
    predictions = np.full(len(labels), -1, dtype=np.int64)
    confidences = np.zeros(len(labels), dtype=np.float32)
    hits = np.zeros(len(labels), dtype=np.int16)
    purities = np.zeros(len(labels), dtype=np.float32)
    fold_reports: List[dict] = []

    for fold in sorted(np.unique(fold_ids).astype(int).tolist()):
        train_idx = np.flatnonzero(fold_ids != fold)
        held_idx = np.flatnonzero(fold_ids == fold)

        print(f"Build V11 retrieval tables for fold {fold}...")
        tables = table_from_indices(
            keys_list,
            labels,
            train_idx,
            specs,
            min_count=min_count,
            max_table_size_per_spec=max_table_size_per_spec,
        )
        table_sizes = {spec: len(table) for spec, table in tables.items()}

        held_keys = [keys_list[int(i)] for i in held_idx]
        fold_prob, fold_pred, fold_conf, fold_hits, fold_purity = predict_retrieval_with_tables(
            held_keys,
            tables,
            specs,
            weights,
            smoothing=smoothing,
        )

        probabilities[held_idx] = fold_prob
        predictions[held_idx] = fold_pred
        confidences[held_idx] = fold_conf
        hits[held_idx] = fold_hits
        purities[held_idx] = fold_purity

        has_hit = fold_hits > 0
        fold_reports.append({
            "fold": int(fold),
            "held": int(len(held_idx)),
            "coverage": float(has_hit.mean()),
            "hit_accuracy": float((fold_pred[has_hit] == labels[held_idx][has_hit]).mean()) if np.any(has_hit) else 0.0,
            "all_macro_f1": macro_f1_fast(labels[held_idx], fold_pred),
            "hit_macro_f1": macro_f1_fast(labels[held_idx][has_hit], fold_pred[has_hit]) if np.any(has_hit) else 0.0,
            "mean_hits": float(fold_hits.mean()),
            "mean_confidence": float(fold_conf.mean()),
            "table_sizes_top": dict(sorted(table_sizes.items(), key=lambda x: x[1], reverse=True)[:10]),
        })
        print(
            f"fold={fold} coverage={fold_reports[-1]['coverage']:.4f} "
            f"hit_acc={fold_reports[-1]['hit_accuracy']:.4f} "
            f"all_macro={fold_reports[-1]['all_macro_f1']:.6f}"
        )

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "confidences": confidences,
        "hits": hits,
        "best_purities": purities,
        "fold_reports": fold_reports,
    }


def config_grid() -> List[dict]:
    configs: List[dict] = [{"strategy": "qwen"}]
    for w_qwen in (0.75, 1.00, 1.25, 1.50, 1.75):
        for w_ret in (0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00):
            for min_hits in (0, 1, 2, 3, 5):
                for min_conf in (0.0, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92):
                    for override_conf in (0.0, 0.80, 0.88, 0.94, 0.98):
                        configs.append({
                            "w_qwen": float(w_qwen),
                            "w_ret": float(w_ret),
                            "min_hits": int(min_hits),
                            "min_conf": float(min_conf),
                            "override_conf": float(override_conf),
                        })
    return configs


def predict_blend(
    config: Mapping[str, Any],
    qwen_probs: np.ndarray,
    retrieval_probs: np.ndarray,
    retrieval_pred: np.ndarray,
    retrieval_conf: np.ndarray,
    retrieval_hits: np.ndarray,
) -> np.ndarray:
    if config.get("strategy") == "qwen":
        return qwen_probs.argmax(axis=1).astype(np.int64)

    active = (
        (retrieval_hits >= int(config["min_hits"]))
        & (retrieval_conf >= float(config["min_conf"]))
    )
    score = np.log(np.maximum(qwen_probs, 1e-12)) * float(config["w_qwen"])

    if np.any(active):
        score[active] += (
            np.log(np.maximum(retrieval_probs[active], 1e-12))
            * float(config["w_ret"])
        )

    pred = score.argmax(axis=1).astype(np.int64)

    override_conf = float(config.get("override_conf", 0.0))
    if override_conf > 0:
        override = active & (retrieval_conf >= override_conf)
        pred[override] = retrieval_pred[override]

    return pred


def nested_select(
    labels: np.ndarray,
    fold_ids: np.ndarray,
    qwen_probs: np.ndarray,
    retrieval: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline = qwen_probs.argmax(axis=1).astype(np.int64)
    baseline_score = macro_f1_fast(labels, baseline)
    unique_folds = sorted(np.unique(fold_ids).astype(int).tolist())
    configs = config_grid()

    ret_probs = retrieval["probabilities"]
    ret_pred = retrieval["predictions"]
    ret_conf = retrieval["confidences"]
    ret_hits = retrieval["hits"]

    pooled_best_score = baseline_score
    pooled_best_config: dict = {"strategy": "qwen"}
    pooled_best_pred = baseline.copy()

    selection_baselines = {
        fold: macro_f1_fast(labels[fold_ids != fold], baseline[fold_ids != fold])
        for fold in unique_folds
    }
    fold_best_scores = dict(selection_baselines)
    fold_best_configs = {fold: {"strategy": "qwen"} for fold in unique_folds}

    print(f"Evaluate {len(configs)} V11 blend configs...")
    for config in tqdm(configs, desc="V11 blend"):
        pred = predict_blend(config, qwen_probs, ret_probs, ret_pred, ret_conf, ret_hits)
        score = macro_f1_fast(labels, pred)

        if score > pooled_best_score:
            pooled_best_score = score
            pooled_best_config = dict(config)
            pooled_best_pred = pred.copy()

        for fold in unique_folds:
            mask = fold_ids != fold
            fold_score = macro_f1_fast(labels[mask], pred[mask])
            if fold_score > fold_best_scores[fold]:
                fold_best_scores[fold] = fold_score
                fold_best_configs[fold] = dict(config)

    raw_nested = baseline.copy()
    fold_records: List[dict] = []
    for fold in unique_folds:
        held = fold_ids == fold
        config = fold_best_configs[fold]
        pred = predict_blend(config, qwen_probs, ret_probs, ret_pred, ret_conf, ret_hits)
        raw_nested[held] = pred[held]
        fold_records.append({
            "fold": int(fold),
            "selection_baseline": float(selection_baselines[fold]),
            "selection_best": float(fold_best_scores[fold]),
            "selection_gain": float(fold_best_scores[fold] - selection_baselines[fold]),
            "raw_config": config,
            "held_samples": int(np.sum(held)),
        })

    raw_nested_score = macro_f1_fast(labels, raw_nested)

    config_counter = Counter(json.dumps(r["raw_config"], sort_keys=True) for r in fold_records)
    consensus_config = json.loads(config_counter.most_common(1)[0][0])
    consensus_pred = predict_blend(consensus_config, qwen_probs, ret_probs, ret_pred, ret_conf, ret_hits)
    consensus_score = macro_f1_fast(labels, consensus_pred)

    return {
        "baseline_score": baseline_score,
        "baseline_predictions": baseline,
        "pooled_score": pooled_best_score,
        "pooled_config": pooled_best_config,
        "pooled_predictions": pooled_best_pred,
        "raw_nested_score": raw_nested_score,
        "raw_nested_predictions": raw_nested,
        "consensus_score": consensus_score,
        "consensus_config": consensus_config,
        "consensus_predictions": consensus_pred,
        "fold_records": fold_records,
    }


def confidence_slices(labels: np.ndarray, pred: np.ndarray, hits: np.ndarray, conf: np.ndarray) -> List[dict]:
    rows: List[dict] = []
    for min_hits in (1, 2, 3, 5, 8):
        for min_conf in (0.45, 0.55, 0.65, 0.75, 0.85, 0.92):
            mask = (hits >= min_hits) & (conf >= min_conf)
            if not np.any(mask):
                continue
            rows.append({
                "min_hits": int(min_hits),
                "min_conf": float(min_conf),
                "coverage": float(mask.mean()),
                "accuracy": float((pred[mask] == labels[mask]).mean()),
                "macro_f1": macro_f1_fast(labels[mask], pred[mask]),
                "samples": int(np.sum(mask)),
            })
    return rows


def build_final_bundle(
    keys_list: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    min_count: int,
    max_table_size_per_spec: int,
) -> Dict[str, Any]:
    weights = spec_weights()
    specs = list(weights.keys())
    tables = table_from_indices(
        keys_list,
        labels,
        np.arange(len(labels), dtype=np.int64),
        specs,
        min_count=min_count,
        max_table_size_per_spec=max_table_size_per_spec,
    )

    # Convert numpy arrays to compact lists for robust joblib transport.
    serializable_tables = {
        spec: {key: vector.astype(np.int16).tolist() for key, vector in table.items()}
        for spec, table in tables.items()
    }

    return {
        "version": "template_sequence_retrieval_v11",
        "classes": ALL_CLASSES,
        "domain_names": DOMAIN_NAMES,
        "domain_to_actions": DOMAIN_TO_ACTIONS,
        "spec_weights": weights,
        "specs": specs,
        "tables": serializable_tables,
        "min_count": int(min_count),
    }


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Load samples and Qwen OOF logits...")
    samples, labels = load_samples_and_labels(args.data, args.labels)
    payload = np.load(args.oof_logits)

    action_logits = payload["action_logits"].astype(np.float32)
    family_logits = payload["family_logits"].astype(np.float32)
    fold_ids = payload["fold_ids"].astype(np.int64)
    oof_labels = payload["labels"].astype(np.int64)

    if not np.array_equal(labels, oof_labels):
        raise RuntimeError("OOF labels do not align with train_labels.csv")

    if args.smoke:
        rng = np.random.default_rng(SEED)
        selected: List[int] = []
        for fold in sorted(np.unique(fold_ids).astype(int).tolist()):
            fold_idx = np.flatnonzero(fold_ids == fold)
            rng.shuffle(fold_idx)
            selected.extend(fold_idx[: min(1200, len(fold_idx))].tolist())
        selected_idx = np.asarray(sorted(selected), dtype=np.int64)
        samples = [samples[int(i)] for i in selected_idx]
        labels = labels[selected_idx]
        action_logits = action_logits[selected_idx]
        family_logits = family_logits[selected_idx]
        fold_ids = fold_ids[selected_idx]
        print("Smoke samples:", len(samples))

    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))
    class_weights = np.asarray(postprocess["training_class_weights"], dtype=np.float64)
    qwen_final_logits = final_logits(action_logits, family_logits, class_weights, postprocess)
    qwen_probs = softmax(qwen_final_logits, axis=1)
    qwen_pred = qwen_probs.argmax(axis=1).astype(np.int64)
    qwen_score = macro_f1_fast(labels, qwen_pred)

    print(f"Qwen baseline OOF Macro-F1: {qwen_score:.6f}")

    signatures, keys_list = make_signatures(samples, qwen_probs)

    retrieval = build_oof_retrieval(
        keys_list,
        labels,
        fold_ids,
        min_count=args.min_count,
        max_table_size_per_spec=args.max_table_size_per_spec,
        smoothing=args.smoothing,
    )
    ret_pred = retrieval["predictions"]
    ret_hits = retrieval["hits"]
    ret_conf = retrieval["confidences"]
    has_hit = ret_hits > 0

    ret_all_score = macro_f1_fast(labels, ret_pred)
    ret_hit_acc = float((ret_pred[has_hit] == labels[has_hit]).mean()) if np.any(has_hit) else 0.0
    ret_hit_macro = macro_f1_fast(labels[has_hit], ret_pred[has_hit]) if np.any(has_hit) else 0.0

    print()
    print(f"Retrieval coverage:      {has_hit.mean():.6f}")
    print(f"Retrieval hit accuracy:  {ret_hit_acc:.6f}")
    print(f"Retrieval all Macro-F1:  {ret_all_score:.6f}")
    print(f"Retrieval hit Macro-F1:  {ret_hit_macro:.6f}")

    selection = nested_select(labels, fold_ids, qwen_probs, retrieval)

    raw_pred = selection["raw_nested_predictions"]
    pooled_pred = selection["pooled_predictions"]
    consensus_pred = selection["consensus_predictions"]

    print()
    print(f"Baseline Macro-F1:       {selection['baseline_score']:.6f}")
    print(f"Pooled Macro-F1:         {selection['pooled_score']:.6f}")
    print(f"Pooled improvement:      {selection['pooled_score'] - selection['baseline_score']:+.6f}")
    print(f"Raw nested Macro-F1:     {selection['raw_nested_score']:.6f}")
    print(f"Raw nested improvement:  {selection['raw_nested_score'] - selection['baseline_score']:+.6f}")
    print(f"Consensus Macro-F1:      {selection['consensus_score']:.6f}")
    print(f"Consensus improvement:   {selection['consensus_score'] - selection['baseline_score']:+.6f}")
    print()
    print("Pooled config:")
    print(json.dumps(selection["pooled_config"], ensure_ascii=False, indent=2))
    print()
    print("Consensus config:")
    print(json.dumps(selection["consensus_config"], ensure_ascii=False, indent=2))

    print()
    print("Class F1 changes, raw nested:")
    base_f1 = class_f1_values(labels, qwen_pred)
    raw_f1 = class_f1_values(labels, raw_pred)
    for i, label in enumerate(ALL_CLASSES):
        print(f"{label:18s} {base_f1[i]:.6f} -> {raw_f1[i]:.6f} ({raw_f1[i] - base_f1[i]:+.6f})")

    report_text = classification_report(
        labels,
        raw_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=ALL_CLASSES,
        digits=6,
        zero_division=0,
    )
    print()
    print(report_text)

    print("Build final retrieval bundle on all samples...")
    final_bundle = build_final_bundle(
        keys_list,
        labels,
        min_count=args.min_count,
        max_table_size_per_spec=args.max_table_size_per_spec,
    )
    final_bundle["pooled_config"] = selection["pooled_config"]
    final_bundle["consensus_config"] = selection["consensus_config"]
    final_bundle["fold_records"] = selection["fold_records"]
    final_bundle["smoothing"] = float(args.smoothing)
    final_bundle["notes"] = (
        "Use with a V4 Qwen inference script: compute qwen probabilities, "
        "build the same V11 signatures/keys for each test sample, retrieve "
        "state-policy probabilities from the saved tables, then blend by "
        "pooled_config or consensus_config."
    )
    joblib.dump(final_bundle, args.output_dir / "template_sequence_retrieval_v11.joblib", compress=3)

    slices = confidence_slices(labels, ret_pred, ret_hits, ret_conf)
    metrics = {
        "samples": int(len(labels)),
        "baseline_oof_macro_f1": float(selection["baseline_score"]),
        "retrieval_coverage": float(has_hit.mean()),
        "retrieval_hit_accuracy": ret_hit_acc,
        "retrieval_all_macro_f1": float(ret_all_score),
        "retrieval_hit_macro_f1": float(ret_hit_macro),
        "pooled_macro_f1": float(selection["pooled_score"]),
        "pooled_improvement": float(selection["pooled_score"] - selection["baseline_score"]),
        "pooled_config": selection["pooled_config"],
        "raw_nested_macro_f1": float(selection["raw_nested_score"]),
        "raw_nested_improvement": float(selection["raw_nested_score"] - selection["baseline_score"]),
        "consensus_macro_f1": float(selection["consensus_score"]),
        "consensus_improvement": float(selection["consensus_score"] - selection["baseline_score"]),
        "consensus_config": selection["consensus_config"],
        "fold_records": selection["fold_records"],
        "retrieval_fold_reports": retrieval["fold_reports"],
        "confidence_slices": slices,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "classification_report_raw_nested.txt").write_text(
        report_text,
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "v11_oof_outputs.npz",
        labels=labels,
        fold_ids=fold_ids,
        qwen_predictions=qwen_pred.astype(np.int8),
        retrieval_predictions=ret_pred.astype(np.int8),
        retrieval_confidences=ret_conf.astype(np.float32),
        retrieval_hits=ret_hits.astype(np.int16),
        pooled_predictions=pooled_pred.astype(np.int8),
        raw_nested_predictions=raw_pred.astype(np.int8),
        consensus_predictions=consensus_pred.astype(np.int8),
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
