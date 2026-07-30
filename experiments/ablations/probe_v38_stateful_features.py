import argparse
import csv
import glob
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


ALL_CLASSES = [
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

ACTION_TO_FAMILY = np.asarray([
    0, 0, 0, 0,
    1, 1, 1,
    2, 2, 2,
    3, 3,
    4,
    3,
], dtype=np.int64)

LABEL_KEYS = ["labels", "y", "targets", "target", "validation_labels", "val_labels"]
INDEX_KEYS = [
    "validation_indices", "val_indices", "valid_indices", "validation_idx",
    "val_idx", "valid_idx", "indices", "idx", "sample_indices", "sample_idx",
]

FEATURE_MODE = "baseline"


CUE_GROUPS = {
    "list_directory": [
        r"뭐가\s*있는지", r"뭐뭐\s*있는지", r"뭐\s*있나", r"뭐\s*있어",
        r"디렉토리", r"폴더", r"루트", r"root", r"repo root", r"directory",
        r"folder", r"tree", r"트리", r"구조부터", r"한눈에", r"목록", r"나열",
        r"under .+? what", r"what'?s sitting", r"what'?s in", r"list .+",
        r"entries", r"files under", r"밑에",
    ],
    "grep_search": [
        r"grep", r"검색", r"찾아", r"찾아봐", r"찾아줘", r"뒤져", r"훑어",
        r"어디서\s*쓰", r"어디에\s*쓰", r"참조", r"사용처", r"들어간\s*데",
        r"전체에서", r"전부\s*찾", r"다\s*찾", r"matches", r"occurrences",
        r"reference", r"references", r"used anywhere", r"where.*used",
        r"find .+ in", r"search .+ for", r"look for",
    ],
    "read_file": [
        r"열어", r"열어서", r"보여", r"보여줘", r"펼쳐", r"내용", r"읽어",
        r"어떻게\s*생겼", r"구현", r"current impl", r"show me", r"open it",
        r"open .+", r"read .+", r"pull up", r"내용\s*좀", r"파일\s*좀",
        r"그\s*부분", r"that file", r"how .+ works today",
    ],
    "glob_pattern": [
        r"\*\*/", r"\*\.py", r"\*\.ts", r"\*\.tsx", r"\*\.js", r"\*\.yaml",
        r"glob", r"패턴", r"확장자", r"파일들", r"전부\s*패턴", r"all .+ files",
        r"every .+ file", r"files matched",
    ],
    "ask_user": [
        r"애매", r"모르겠", r"헷갈", r"어느\s*쪽", r"둘\s*중", r"뭘\s*원",
        r"뭐부터", r"정해줘야", r"물어봐", r"나한테", r"선호", r"요구가",
        r"vague", r"not sure", r"don'?t know", r"i genuinely don'?t know",
        r"which .* should", r"should .* or", r"do you think", r"what would you ask",
        r"pin it down", r"unclear",
    ],
    "plan_task": [
        r"단계", r"쪼개", r"순서", r"계획", r"진행", r"작업\s*범위",
        r"plan", r"plan it out", r"walk me through", r"approach", r"break .* into",
        r"split .* steps", r"step by step", r"how would you approach", r"roadmap",
        r"먼저\s*어떻게", r"어디부터\s*어떻게",
    ],
    "run_bash": [
        r"실행", r"돌려", r"띄워", r"빌드", r"다시\s*돌", r"한번\s*돌",
        r"run ", r"rerun", r"build", r"smoke", r"dry-?run", r"start", r"serve",
        r"docker build", r"npm run", r"pytest", r"go test", r"go build", r"cargo run",
        r"python manage\.py", r"uvicorn", r"mvn", r"gradlew", r"exit=", r"stderr",
    ],
    "run_tests": [
        r"테스트", r"test", r"tests", r"pytest", r"jest", r"vitest", r"go test",
        r"cargo test", r"suite", r"통과", r"green", r"pass", r"failing",
    ],
    "lint_or_typecheck": [
        r"lint", r"린트", r"typecheck", r"type check", r"타입\s*체크", r"정적분석",
        r"tsc\s+--noemit", r"ruff", r"mypy", r"go vet", r"cargo check",
        r"eslint", r"pyright", r"no issues", r"errors?, \d+ files affected",
    ],
    "web_search": [
        r"공식\s*문서", r"문서\s*찾", r"검색해", r"찾아봐", r"요즘\s*권장",
        r"최신", r"current recommended", r"recommended", r"docs", r"documentation",
        r"look up", r"google", r"web", r"pypi", r"version",
    ],
    "edit_file": [
        r"고쳐", r"수정", r"바꿔", r"추가해", r"넣어", r"패치", r"손봐",
        r"edit", r"modify", r"change", r"fix", r"add ", r"wire", r"update",
    ],
    "write_file": [
        r"새로\s*만들", r"통째로\s*다시", r"갈아엎", r"rewrite", r"create .* file",
        r"new file", r"write .* from scratch",
    ],
    "apply_patch": [
        r"한꺼번에", r"두\s*파일", r"여러\s*파일", r"n_files", r"patch", r"패치",
        r"같이\s*맞춰", r"둘\s*다", r"전반",
    ],
}


def macro_f1(y, pred):
    return float(f1_score(y, pred, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0))


def class_f1(y, pred):
    return f1_score(y, pred, labels=np.arange(len(ALL_CLASSES)), average=None, zero_division=0)


def softmax(logits):
    x = logits.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def log_softmax(logits):
    p = softmax(logits)
    return np.log(np.maximum(p, 1e-12))


def final_logits(action_logits, family_logits, postprocess, cfg):
    out = action_logits.astype(np.float64) / float(cfg.get("action_temperature", 1.0))
    family_weight = float(cfg.get("family_weight", 0.0))
    prior_beta = float(cfg.get("prior_beta", 0.0))

    if family_logits is not None and family_weight != 0:
        out = out + family_weight * family_logits.astype(np.float64)[:, ACTION_TO_FAMILY]

    if prior_beta != 0:
        class_weights = np.asarray(
            postprocess.get("training_class_weights", np.ones(len(ALL_CLASSES))),
            dtype=np.float64,
        )
        out = out - prior_beta * np.log(np.maximum(class_weights, 1e-12))[None, :]

    return out


def tune_qwen(y, q_npz, postprocess):
    action_logits = q_npz["action_logits"]
    family_logits = q_npz["family_logits"] if "family_logits" in q_npz.files else None

    configs = [{"action_temperature": 1.0, "family_weight": 0.0, "prior_beta": 0.0}]
    for t in [0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0]:
        for fw in [0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.25]:
            for pb in [-1.5, -1.0, -0.75, -0.5, 0.0, 0.25, 0.5]:
                configs.append({"action_temperature": t, "family_weight": fw, "prior_beta": pb})

    best_score = -1.0
    best_logits = None
    best_cfg = None
    for cfg in configs:
        logits = final_logits(action_logits, family_logits, postprocess, cfg)
        pred = logits.argmax(axis=1)
        score = macro_f1(y, pred)
        if score > best_score:
            best_score = score
            best_logits = logits
            best_cfg = cfg

    return best_logits, best_cfg, best_score


def first_key(npz, keys):
    for k in keys:
        if k in npz.files:
            return k
    return None


def load_labels_csv(path):
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No rows in labels csv: {path}")

    preferred = ["action", "label", "labels", "target", "class", "answer", "y"]
    cols = [c for c in preferred if c in rows[0]]
    cols += [c for c in fieldnames if c not in cols]

    def norm(v):
        s = str(v).strip()
        if s.isdigit():
            return int(s)
        if s in ALL_CLASSES:
            return ALL_CLASSES.index(s)
        return None

    best_col = None
    best_vals = None
    best_ok = -1
    for col in cols:
        vals = []
        ok = 0
        for r in rows:
            v = norm(r.get(col, ""))
            vals.append(v)
            if v is not None and 0 <= v < len(ALL_CLASSES):
                ok += 1
        if ok > best_ok:
            best_col = col
            best_vals = vals
            best_ok = ok

    if best_ok != len(rows):
        raise RuntimeError(f"Could not parse labels: best_col={best_col}, ok={best_ok}/{len(rows)}")

    return np.asarray(best_vals, dtype=np.int64), best_col


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append({"raw": line})
    return rows


def compact(obj, max_chars=8000):
    if obj is None:
        return ""
    if isinstance(obj, str):
        s = obj
    else:
        try:
            s = json.dumps(obj, ensure_ascii=False)
        except Exception:
            s = str(obj)
    s = s.replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    if len(s) > max_chars:
        s = s[:max_chars] + " ..."
    return s


def flatten_action(a):
    if not isinstance(a, dict):
        return compact(a)
    parts = []
    name = a.get("name")
    if name:
        parts.append(f"assistant_action_name={name}")
    args = a.get("args")
    if args:
        parts.append(f"args={compact(args, 1000)}")
    rs = a.get("result_summary")
    if rs:
        parts.append(f"result={compact(rs, 1000)}")
    return " ".join(parts)


def extract_current_prompt(row):
    if not isinstance(row, dict):
        return compact(row)

    for key in ["current_prompt", "current", "prompt", "query", "instruction", "input", "content", "text"]:
        if key in row and row[key] not in (None, "", []):
            return compact(row[key])

    return ""


def extract_text(row):
    if not isinstance(row, dict):
        return compact(row)

    parts = []

    cur = extract_current_prompt(row)
    if cur:
        parts.append(f"CURRENT: {cur}")

    hist = row.get("history")
    if isinstance(hist, list):
        for item in hist[-8:]:
            if isinstance(item, dict):
                role = item.get("role", "")
                if role == "assistant_action":
                    parts.append(flatten_action(item))
                else:
                    content = item.get("content", "")
                    if content:
                        parts.append(f"{role}: {compact(content)}")
            else:
                parts.append(compact(item))
    elif hist:
        parts.append(f"HISTORY: {compact(hist)}")

    # Include useful metadata lightly.
    meta = row.get("session_meta")
    if isinstance(meta, dict):
        ws = meta.get("workspace", {})
        if isinstance(ws, dict):
            parts.append(f"workspace={compact(ws, 800)}")
        lp = meta.get("language_pref")
        if lp:
            parts.append(f"language_pref={lp}")

    if not parts:
        parts.append(compact(row))

    return "\n".join(parts)


def extract_current_only_text(row):
    cur = extract_current_prompt(row)
    return cur if cur else extract_text(row)


def get_last_action(row):
    if not isinstance(row, dict):
        return ""
    hist = row.get("history")
    if not isinstance(hist, list):
        return ""
    for item in reversed(hist):
        if isinstance(item, dict) and item.get("role") == "assistant_action":
            return str(item.get("name", ""))
    return ""


def count_regex(patterns, text):
    if not text:
        return 0
    return sum(1 for p in patterns if re.search(p, text, flags=re.IGNORECASE))


def structured_features(rows, texts, current_texts):
    feats = []
    last_actions = []

    path_re = re.compile(r"[\w./\\-]+\.(py|ts|tsx|js|jsx|rs|go|java|kt|yaml|yml|json|toml|sh|md|txt|tf|sql|vue|tsx|css)", re.I)
    dir_re = re.compile(r"(src|app|lib|config|configs|scripts|tests|components|routes|models|data|dags|terraform|k8s|\.github|ios|android)(/|\\|\s|$)", re.I)
    cmd_re = re.compile(r"(npm|yarn|pnpm|pytest|python|go|cargo|docker|mvn|gradlew|ruff|tsc|vitest|jest|uvicorn|bash)\s+", re.I)

    for row, text, cur in zip(rows, texts, current_texts):
        tl = text.lower()
        cl = cur.lower()
        last = get_last_action(row)
        last_actions.append(last)

        f = []
        f.append(len(text))
        f.append(len(cur))
        f.append(text.count("\n"))
        f.append(cur.count("?") + cur.count("？"))
        f.append(cur.count("!") + cur.count("！"))
        f.append(len(path_re.findall(text)))
        f.append(len(path_re.findall(cur)))
        f.append(len(dir_re.findall(text)))
        f.append(len(dir_re.findall(cur)))
        f.append(len(cmd_re.findall(text)))
        f.append(len(cmd_re.findall(cur)))
        f.append(int("?" in cur))
        f.append(int("ㅠ" in cur or "ㅜ" in cur))
        f.append(int("..." in cur or "…" in cur))
        f.append(int("ERROR" in text or "error" in text))
        f.append(int("PASS" in text or "passed" in tl or "green" in tl))
        f.append(int("FAIL" in text or "failed" in tl))
        f.append(int("permission denied" in tl))
        f.append(int("target_symbol" in tl))
        f.append(int("result_summary" in tl))

        for cls in ALL_CLASSES:
            patterns = CUE_GROUPS.get(cls, [])
            f.append(count_regex(patterns, cur))
            f.append(count_regex(patterns, text))
            f.append(int(count_regex(patterns, cur) > 0))
            f.append(int(count_regex(patterns, text) > 0))

        # Confusion-specific differences.
        f.append(count_regex(CUE_GROUPS["list_directory"], cur) - count_regex(CUE_GROUPS["read_file"], cur))
        f.append(count_regex(CUE_GROUPS["grep_search"], cur) - count_regex(CUE_GROUPS["read_file"], cur))
        f.append(count_regex(CUE_GROUPS["ask_user"], cur) - count_regex(CUE_GROUPS["plan_task"], cur))
        f.append(count_regex(CUE_GROUPS["lint_or_typecheck"], cur) - count_regex(CUE_GROUPS["run_bash"], cur))
        f.append(count_regex(CUE_GROUPS["web_search"], cur) - count_regex(CUE_GROUPS["grep_search"], cur))

        feats.append(f)

    X = np.asarray(feats, dtype=np.float32)

    # Last action one-hot.
    last_onehot = np.zeros((len(rows), len(ALL_CLASSES) + 1), dtype=np.float32)
    action_to_id = {a: i for i, a in enumerate(ALL_CLASSES)}
    for i, a in enumerate(last_actions):
        last_onehot[i, action_to_id.get(a, len(ALL_CLASSES))] = 1.0

    parts = [X, last_onehot]
    if FEATURE_MODE in {"action8", "combined"}:
        parts.append(action8_state_features(rows))
    if FEATURE_MODE in {"openfiles", "combined"}:
        parts.append(open_file_state_features(rows, current_texts))
    return np.hstack(parts)



FILE_EXTENSIONS = [
    "py", "js", "ts", "tsx", "jsx", "java", "go", "rs", "cpp", "c", "h",
    "cs", "json", "yaml", "yml", "toml", "ini", "md", "txt", "sh", "ps1",
    "sql", "html", "css", "vue", "svelte", "ipynb",
]

CONFIG_BASENAMES = {
    "package.json", "pyproject.toml", "requirements.txt", "setup.py",
    "setup.cfg", "tox.ini", "tsconfig.json", "eslint.config.js",
    ".eslintrc", ".eslintrc.json", "dockerfile", "docker-compose.yml",
    "cargo.toml", "go.mod", "pom.xml", "build.gradle", "gradlew",
}

ACTION_TO_ID = {name: index for index, name in enumerate(ALL_CLASSES)}
NONE_ACTION_ID = len(ALL_CLASSES)


def get_action_history(row, limit=8):
    if not isinstance(row, dict):
        return []
    history = row.get("history", [])
    if not isinstance(history, list):
        return []

    actions = []
    for item in history:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "assistant_action" or item.get("name"):
            name = str(item.get("name", "")).strip()
            if name:
                actions.append(name)
    return actions[-limit:]


def get_open_files(row):
    if not isinstance(row, dict):
        return []
    meta = row.get("session_meta", {})
    if not isinstance(meta, dict):
        return []
    workspace = meta.get("workspace", {})
    if not isinstance(workspace, dict):
        return []
    values = workspace.get("open_files", [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if value not in (None, "")]


def normalize_path(path):
    return str(path).replace("\\", "/").lower().strip()


def path_basename(path):
    return normalize_path(path).rsplit("/", 1)[-1]


def path_extension(path):
    name = path_basename(path)
    return name.rsplit(".", 1)[-1] if "." in name else ""


def action8_state_features(rows):
    positional = np.zeros(
        (len(rows), 8 * (len(ALL_CLASSES) + 1)),
        dtype=np.float32,
    )
    pair_onehot = np.zeros(
        (len(rows), (len(ALL_CLASSES) + 1) ** 2),
        dtype=np.float32,
    )
    numeric = []

    for row_index, row in enumerate(rows):
        actions = get_action_history(row, limit=8)
        padded = [None] * (8 - len(actions)) + actions

        for position, action in enumerate(padded):
            action_id = ACTION_TO_ID.get(action, NONE_ACTION_ID)
            positional[
                row_index,
                position * (len(ALL_CLASSES) + 1) + action_id,
            ] = 1.0

        if len(actions) >= 2:
            left_id = ACTION_TO_ID.get(actions[-2], NONE_ACTION_ID)
            right_id = ACTION_TO_ID.get(actions[-1], NONE_ACTION_ID)
            pair_onehot[
                row_index,
                left_id * (len(ALL_CLASSES) + 1) + right_id,
            ] = 1.0

        switches = sum(
            actions[index] != actions[index - 1]
            for index in range(1, len(actions))
        )
        repeat_last = (
            sum(action == actions[-1] for action in actions)
            if actions else 0
        )

        transitions = [
            sum(
                left == "list_directory" and right == "read_file"
                for left, right in zip(actions[:-1], actions[1:])
            ),
            sum(
                left == "grep_search" and right == "read_file"
                for left, right in zip(actions[:-1], actions[1:])
            ),
            sum(
                left == "read_file"
                and right in {"edit_file", "write_file", "apply_patch"}
                for left, right in zip(actions[:-1], actions[1:])
            ),
            sum(
                left in {"edit_file", "write_file", "apply_patch"}
                and right == "run_tests"
                for left, right in zip(actions[:-1], actions[1:])
            ),
            sum(
                left == "run_tests"
                and right in {"edit_file", "apply_patch"}
                for left, right in zip(actions[:-1], actions[1:])
            ),
            sum(
                left == "run_bash" and right == "lint_or_typecheck"
                for left, right in zip(actions[:-1], actions[1:])
            ),
            sum(
                left == "ask_user" and right == "plan_task"
                for left, right in zip(actions[:-1], actions[1:])
            ),
        ]

        numeric.append([
            len(actions) / 8.0,
            switches / 7.0,
            repeat_last / 8.0,
            int(len(actions) >= 2 and len(set(actions)) == 1),
            *[min(value, 3) / 3.0 for value in transitions],
        ])

    return np.hstack([
        positional,
        pair_onehot,
        np.asarray(numeric, dtype=np.float32),
    ])


def open_file_state_features(rows, current_texts):
    features = []

    for row, current in zip(rows, current_texts):
        files = [normalize_path(path) for path in get_open_files(row)]
        names = [path_basename(path) for path in files]
        extensions = [path_extension(path) for path in files]
        current_lower = current.lower()

        extension_counts = [
            min(extensions.count(ext), 5) / 5.0
            for ext in FILE_EXTENSIONS
        ]

        test_count = sum(
            "/test/" in path
            or "/tests/" in path
            or path_basename(path).startswith("test_")
            or path_basename(path).endswith("_test.py")
            or ".spec." in path_basename(path)
            or ".test." in path_basename(path)
            for path in files
        )
        config_count = sum(
            path_basename(path) in CONFIG_BASENAMES
            or "/config/" in path
            or "/configs/" in path
            for path in files
        )
        source_count = sum(
            any(marker in path for marker in ["/src/", "/app/", "/lib/"])
            for path in files
        )
        documentation_count = sum(
            path_extension(path) == "md"
            or path_basename(path).startswith("readme")
            for path in files
        )
        depths = [path.count("/") for path in files]

        exact_name_mentions = sum(
            bool(name) and name in current_lower
            for name in names
        )
        extension_mentions = sum(
            bool(ext)
            and re.search(rf"\b{re.escape(ext)}\b", current_lower)
            is not None
            for ext in set(extensions)
        )

        features.append([
            min(len(files), 12) / 12.0,
            min(test_count, 5) / 5.0,
            min(config_count, 5) / 5.0,
            min(source_count, 8) / 8.0,
            min(documentation_count, 5) / 5.0,
            (float(np.mean(depths)) / 10.0) if depths else 0.0,
            (float(np.max(depths)) / 15.0) if depths else 0.0,
            len(set(extensions)) / max(len(extensions), 1),
            int(test_count > 0),
            int(config_count > 0),
            int("package.json" in names),
            int("pyproject.toml" in names),
            int("requirements.txt" in names),
            int("tsconfig.json" in names),
            int(any(name.startswith("readme") for name in names)),
            int("ipynb" in extensions),
            int(exact_name_mentions > 0),
            min(exact_name_mentions, 5) / 5.0,
            min(extension_mentions, 5) / 5.0,
            int(re.search(r"\*\*?/|\*\.[a-z0-9]+", current_lower) is not None),
            *extension_counts,
        ])

    return np.asarray(features, dtype=np.float32)

def load_or_find_val_indices(args, val_y, full_y):
    if args.val_indices:
        p = Path(args.val_indices)
        if p.suffix.lower() == ".npy":
            arr = np.load(p, allow_pickle=False)
        elif p.suffix.lower() == ".npz":
            z = np.load(p, allow_pickle=False)
            key = first_key(z, INDEX_KEYS)
            if key is None:
                raise RuntimeError(f"No validation index key in {p}. keys={z.files}")
            arr = z[key]
        else:
            arr = np.asarray([int(x.strip().split(",")[0]) for x in p.read_text().splitlines() if x.strip()])
        arr = np.asarray(arr).reshape(-1).astype(np.int64)
        if len(arr) != len(val_y):
            raise RuntimeError(f"val index length mismatch: {len(arr)} vs {len(val_y)}")
        if not np.array_equal(full_y[arr], val_y):
            raise RuntimeError("Provided val indices do not match qwen validation labels.")
        return arr, {"source": str(p)}

    # Search for exact validation_indices arrays from previous npz files.
    hits = []
    for pat in args.index_search_glob:
        for item in glob.glob(pat, recursive=True):
            p = Path(item)
            if not p.is_file() or p.suffix.lower() != ".npz":
                continue
            try:
                z = np.load(p, allow_pickle=False)
            except Exception:
                continue
            for key in INDEX_KEYS:
                if key not in z.files:
                    continue
                arr = np.asarray(z[key]).reshape(-1)
                if len(arr) != len(val_y) or not np.issubdtype(arr.dtype, np.integer):
                    continue
                arr = arr.astype(np.int64)
                if arr.min(initial=0) < 0 or arr.max(initial=0) >= len(full_y):
                    continue
                if np.array_equal(full_y[arr], val_y):
                    score = sum(tok in str(p).lower() for tok in ["v4", "validation", "logits", "qwen"])
                    hits.append((score, str(p), key, arr))

    hits.sort(reverse=True, key=lambda x: x[0])
    if not hits:
        raise RuntimeError("Could not find validation indices. Pass --val-indices explicitly.")

    score, path, key, arr = hits[0]
    return arr, {"source": path, "key": key}


def get_model(model_name, args, n_classes):
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as e:
            raise RuntimeError(
                "lightgbm is not installed. Install with: pip install lightgbm"
            ) from e

        return LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            class_weight="balanced" if args.class_weight_balanced else None,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbose=-1,
        )

    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except Exception as e:
            raise RuntimeError(
                "catboost is not installed. Install with: pip install catboost"
            ) from e

        return CatBoostClassifier(
            loss_function="MultiClass",
            iterations=args.n_estimators,
            learning_rate=args.learning_rate,
            depth=args.cat_depth,
            l2_leaf_reg=args.reg_lambda,
            random_seed=args.seed,
            auto_class_weights="Balanced" if args.class_weight_balanced else None,
            allow_writing_files=False,
            verbose=100,
            thread_count=args.n_jobs,
        )

    if model_name == "sgd":
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(
            loss="log_loss",
            alpha=args.sgd_alpha,
            max_iter=args.sgd_max_iter,
            class_weight="balanced" if args.class_weight_balanced else None,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )

    raise ValueError(f"unknown model: {model_name}")


def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if isinstance(p, list):
            p = np.vstack(p).T
        p = np.asarray(p, dtype=np.float64)
        if p.shape[1] != len(ALL_CLASSES):
            # Some classifiers may drop classes.
            out = np.zeros((p.shape[0], len(ALL_CLASSES)), dtype=np.float64)
            classes = getattr(model, "classes_", np.arange(p.shape[1]))
            for j, cls in enumerate(classes):
                if int(cls) < len(ALL_CLASSES):
                    out[:, int(cls)] = p[:, j]
            p = out
        p = np.maximum(p, 1e-12)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    if hasattr(model, "decision_function"):
        logits = model.decision_function(X)
        return softmax(logits)

    pred = model.predict(X)
    p = np.full((len(pred), len(ALL_CLASSES)), 1e-6, dtype=np.float64)
    p[np.arange(len(pred)), pred.astype(int)] = 1.0
    p /= p.sum(axis=1, keepdims=True)
    return p


def blend_search(y, q_logits, tree_prob):
    q_lp = log_softmax(q_logits)
    t_lp = np.log(np.maximum(tree_prob, 1e-12))

    rows = []
    best = {"score": macro_f1(y, q_logits.argmax(axis=1)), "weight_tree": 0.0, "pred": q_logits.argmax(axis=1)}

    for w in np.linspace(0.0, 1.0, 41):
        lp = (1.0 - w) * q_lp + w * t_lp
        pred = lp.argmax(axis=1)
        score = macro_f1(y, pred)
        rows.append({"weight_tree": float(w), "score": float(score), "gain_over_qwen": float(score - best["score"])})
        if score > best["score"]:
            best = {"score": float(score), "weight_tree": float(w), "pred": pred}

    return best, rows


def multiseed_blend_holdout(y, q_logits, tree_prob, seeds, holdout_ratio):
    q_base_pred = q_logits.argmax(axis=1)
    q_lp = log_softmax(q_logits)
    t_lp = np.log(np.maximum(tree_prob, 1e-12))

    rows = []
    for seed in seeds:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=holdout_ratio, random_state=seed)
        tr, ho = next(splitter.split(np.zeros(len(y)), y))

        best_w = 0.0
        best_train = macro_f1(y[tr], q_base_pred[tr])
        for w in np.linspace(0.0, 1.0, 41):
            pred = ((1.0 - w) * q_lp[tr] + w * t_lp[tr]).argmax(axis=1)
            score = macro_f1(y[tr], pred)
            if score > best_train:
                best_train = score
                best_w = float(w)

        full_pred = ((1.0 - best_w) * q_lp + best_w * t_lp).argmax(axis=1)
        row = {
            "seed": int(seed),
            "best_weight_tree": float(best_w),
            "train_qwen": macro_f1(y[tr], q_base_pred[tr]),
            "train_blend": macro_f1(y[tr], full_pred[tr]),
            "train_gain": macro_f1(y[tr], full_pred[tr]) - macro_f1(y[tr], q_base_pred[tr]),
            "holdout_qwen": macro_f1(y[ho], q_base_pred[ho]),
            "holdout_blend": macro_f1(y[ho], full_pred[ho]),
            "holdout_gain": macro_f1(y[ho], full_pred[ho]) - macro_f1(y[ho], q_base_pred[ho]),
            "all_qwen": macro_f1(y, q_base_pred),
            "all_blend": macro_f1(y, full_pred),
            "all_gain": macro_f1(y, full_pred) - macro_f1(y, q_base_pred),
        }
        rows.append(row)
    return rows


def complementarity(y, q_pred, t_pred):
    q_ok = q_pred == y
    t_ok = t_pred == y
    oracle = q_pred.copy()
    oracle[(~q_ok) & t_ok] = t_pred[(~q_ok) & t_ok]
    return {
        "qwen_correct_tree_correct": int((q_ok & t_ok).sum()),
        "qwen_correct_tree_wrong": int((q_ok & (~t_ok)).sum()),
        "qwen_wrong_tree_correct": int(((~q_ok) & t_ok).sum()),
        "qwen_wrong_tree_wrong": int(((~q_ok) & (~t_ok)).sum()),
        "oracle_score": macro_f1(y, oracle),
    }


def write_csv(rows, path):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, key):
    arr = np.asarray([r[key] for r in rows], dtype=np.float64)
    return {
        f"{key}_mean": float(arr.mean()),
        f"{key}_min": float(arr.min()),
        f"{key}_max": float(arr.max()),
        f"{key}_positive": int((arr > 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--qwen-logits", type=Path, required=True)
    ap.add_argument("--postprocess", type=Path, required=True)
    ap.add_argument("--val-indices", type=Path, default=None)
    ap.add_argument("--index-search-glob", action="append", default=["*.npz", "model/**/*.npz"])
    ap.add_argument("--output-dir", type=Path, default=Path("model/v23_tree_probe"))
    ap.add_argument("--feature-mode", choices=["baseline", "action8", "openfiles", "combined"], default="baseline")

    ap.add_argument("--model", choices=["lightgbm", "catboost", "sgd"], default="lightgbm")
    ap.add_argument("--word-features", type=int, default=30000)
    ap.add_argument("--char-features", type=int, default=30000)
    ap.add_argument("--word-ngram-max", type=int, default=3)
    ap.add_argument("--char-ngram-min", type=int, default=3)
    ap.add_argument("--char-ngram-max", type=int, default=5)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=0)

    ap.add_argument("--n-estimators", type=int, default=450)
    ap.add_argument("--learning-rate", type=float, default=0.045)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--max-depth", type=int, default=-1)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample-bytree", type=float, default=0.65)
    ap.add_argument("--reg-alpha", type=float, default=0.0)
    ap.add_argument("--reg-lambda", type=float, default=1.5)
    ap.add_argument("--cat-depth", type=int, default=6)
    ap.add_argument("--sgd-alpha", type=float, default=1e-5)
    ap.add_argument("--sgd-max-iter", type=int, default=50)
    ap.add_argument("--class-weight-balanced", action="store_true", default=True)
    ap.add_argument("--no-class-weight-balanced", action="store_false", dest="class_weight_balanced")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--holdout-ratio", type=float, default=0.35)
    ap.add_argument("--seeds", default="11,22,33,42,55,66,77,88,99,123")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    global FEATURE_MODE
    FEATURE_MODE = args.feature_mode
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Feature mode: {FEATURE_MODE}")
    print("Load labels/logits...")
    q_npz = np.load(args.qwen_logits, allow_pickle=False)
    label_key = first_key(q_npz, LABEL_KEYS)
    if label_key is None:
        raise RuntimeError(f"qwen logits has no label key: keys={q_npz.files}")
    val_y = q_npz[label_key].astype(np.int64)

    full_y, label_col = load_labels_csv(args.labels_csv)
    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))

    q_logits, q_cfg, q_score = tune_qwen(val_y, q_npz, postprocess)
    q_pred = q_logits.argmax(axis=1)

    print(f"Qwen tuned Macro-F1: {q_score:.6f}")
    print(f"Qwen cfg: {json.dumps(q_cfg, ensure_ascii=False)}")

    val_idx, val_meta = load_or_find_val_indices(args, val_y, full_y)
    train_mask = np.ones(len(full_y), dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.where(train_mask)[0]

    if args.max_train and args.max_train > 0 and args.max_train < len(train_idx):
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(train_idx, size=args.max_train, replace=False)
        train_idx = np.sort(train_idx)

    print()
    print("Split:")
    print(f"  full rows:  {len(full_y)}")
    print(f"  train rows: {len(train_idx)}")
    print(f"  val rows:   {len(val_idx)}")
    print(f"  val source: {json.dumps(val_meta, ensure_ascii=False)}")
    print(f"  label col:  {label_col}")

    rows = load_jsonl(args.data)
    if len(rows) != len(full_y):
        raise RuntimeError(f"data/label row mismatch: {len(rows)} vs {len(full_y)}")

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    train_y = full_y[train_idx]

    print()
    print("Extract text...")
    train_texts = [extract_text(r) for r in train_rows]
    val_texts = [extract_text(r) for r in val_rows]
    train_current = [extract_current_only_text(r) for r in train_rows]
    val_current = [extract_current_only_text(r) for r in val_rows]

    print("Vectorize TF-IDF...")
    word_vec = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        ngram_range=(1, args.word_ngram_max),
        max_features=args.word_features,
        min_df=args.min_df,
        sublinear_tf=True,
        strip_accents=None,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=(args.char_ngram_min, args.char_ngram_max),
        max_features=args.char_features,
        min_df=args.min_df,
        sublinear_tf=True,
    )

    Xw_tr = word_vec.fit_transform(train_texts)
    Xw_va = word_vec.transform(val_texts)
    Xc_tr = char_vec.fit_transform(train_current)
    Xc_va = char_vec.transform(val_current)

    print("Build structured cue features...")
    Xs_tr_dense = structured_features(train_rows, train_texts, train_current)
    Xs_va_dense = structured_features(val_rows, val_texts, val_current)
    scaler = StandardScaler()
    Xs_tr = sparse.csr_matrix(scaler.fit_transform(Xs_tr_dense))
    Xs_va = sparse.csr_matrix(scaler.transform(Xs_va_dense))

    X_tr = sparse.hstack([Xw_tr, Xc_tr, Xs_tr], format="csr")
    X_va = sparse.hstack([Xw_va, Xc_va, Xs_va], format="csr")

    print()
    print("Feature matrix:")
    print(f"  X_train: {X_tr.shape}, nnz={X_tr.nnz}")
    print(f"  X_val:   {X_va.shape}, nnz={X_va.nnz}")

    print()
    print(f"Train {args.model}...")
    model = get_model(args.model, args, len(ALL_CLASSES))
    model.fit(X_tr, train_y)

    tree_prob = safe_predict_proba(model, X_va)
    tree_pred = tree_prob.argmax(axis=1)
    tree_score = macro_f1(val_y, tree_pred)

    comp = complementarity(val_y, q_pred, tree_pred)
    comp["oracle_gain_over_qwen"] = comp["oracle_score"] - q_score

    print()
    print("V23 tree diagnostics:")
    print(f"  Qwen Macro-F1: {q_score:.6f}")
    print(f"  Tree Macro-F1: {tree_score:.6f}")
    print(f"  Qwen wrong / Tree correct: {comp['qwen_wrong_tree_correct']}")
    print(f"  Qwen correct / Tree wrong: {comp['qwen_correct_tree_wrong']}")
    print(f"  Oracle Macro-F1: {comp['oracle_score']:.6f}")
    print(f"  Oracle gain: {comp['oracle_gain_over_qwen']:+.6f}")

    blend_best, blend_rows = blend_search(val_y, q_logits, tree_prob)
    print()
    print("Full validation blend search:")
    print(f"  best weight_tree: {blend_best['weight_tree']:.3f}")
    print(f"  blend Macro-F1:   {blend_best['score']:.6f}")
    print(f"  blend gain:       {blend_best['score'] - q_score:+.6f}")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    holdout_rows = multiseed_blend_holdout(val_y, q_logits, tree_prob, seeds, args.holdout_ratio)
    hold_summary = {}
    hold_summary.update(summarize(holdout_rows, "train_gain"))
    hold_summary.update(summarize(holdout_rows, "holdout_gain"))
    hold_summary.update(summarize(holdout_rows, "all_gain"))

    print()
    print("Multi-seed blend holdout summary:")
    print(
        f"  train gain mean:    {hold_summary['train_gain_mean']:+.6f} / "
        f"min {hold_summary['train_gain_min']:+.6f} / max {hold_summary['train_gain_max']:+.6f}"
    )
    print(
        f"  holdout gain mean:  {hold_summary['holdout_gain_mean']:+.6f} / "
        f"min {hold_summary['holdout_gain_min']:+.6f} / max {hold_summary['holdout_gain_max']:+.6f}"
    )
    print(f"  holdout positive:   {hold_summary['holdout_gain_positive']} / {len(holdout_rows)}")
    print(
        f"  all gain mean:      {hold_summary['all_gain_mean']:+.6f} / "
        f"min {hold_summary['all_gain_min']:+.6f} / max {hold_summary['all_gain_max']:+.6f}"
    )

    if args.report:
        print()
        print("Classification report for tree:")
        print(classification_report(
            val_y,
            tree_pred,
            labels=np.arange(len(ALL_CLASSES)),
            target_names=ALL_CLASSES,
            digits=6,
            zero_division=0,
        ))

        print()
        print("Class F1 Qwen -> Tree:")
        q_f1 = class_f1(val_y, q_pred)
        t_f1 = class_f1(val_y, tree_pred)
        for i, cls in enumerate(ALL_CLASSES):
            print(f"{cls:18s} {q_f1[i]:.6f} -> {t_f1[i]:.6f} ({t_f1[i]-q_f1[i]:+.6f})")

        print()
        print("Class F1 Qwen -> Full blend:")
        b_f1 = class_f1(val_y, blend_best["pred"])
        for i, cls in enumerate(ALL_CLASSES):
            print(f"{cls:18s} {q_f1[i]:.6f} -> {b_f1[i]:.6f} ({b_f1[i]-q_f1[i]:+.6f})")

    # Save outputs.
    write_csv(blend_rows, args.output_dir / "blend_search.csv")
    write_csv(holdout_rows, args.output_dir / "multiseed_blend_holdout.csv")
    np.save(args.output_dir / "pred_qwen.npy", q_pred.astype(np.int64))
    np.save(args.output_dir / "pred_tree.npy", tree_pred.astype(np.int64))
    np.save(args.output_dir / "pred_blend_best.npy", blend_best["pred"].astype(np.int64))
    np.save(args.output_dir / "tree_prob_val.npy", tree_prob.astype(np.float32))
    np.save(args.output_dir / "val_indices.npy", val_idx.astype(np.int64))

    summary = {
        "feature_mode": FEATURE_MODE,
        "model": args.model,
        "qwen_score": q_score,
        "qwen_cfg": q_cfg,
        "tree_score": tree_score,
        "complementarity": comp,
        "blend_best": {
            "weight_tree": blend_best["weight_tree"],
            "score": blend_best["score"],
            "gain_over_qwen": blend_best["score"] - q_score,
        },
        "holdout_summary": hold_summary,
        "val_meta": val_meta,
        "feature_shape_train": list(X_tr.shape),
        "feature_shape_val": list(X_va.shape),
        "elapsed_sec": time.time() - start,
        "args": vars(args),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print(f"  {args.output_dir / 'summary.json'}")
    print(f"  {args.output_dir / 'blend_search.csv'}")
    print(f"  {args.output_dir / 'multiseed_blend_holdout.csv'}")
    print(f"  {args.output_dir / 'tree_prob_val.npy'}")
    print(f"  {args.output_dir / 'val_indices.npy'}")

    if hold_summary["holdout_gain_mean"] > 0.0005 and hold_summary["holdout_gain_positive"] >= math.ceil(0.7 * len(holdout_rows)):
        print()
        print("Decision hint: v23 has robust blend signal. Consider constrained submit integration.")
    elif tree_score >= 0.72 and comp["qwen_wrong_tree_correct"] >= 350:
        print()
        print("Decision hint: tree has complementarity but blend is not robust. Consider pair-specific selector probe, not direct blend.")
    else:
        print()
        print("Decision hint: v23 tree is not strong enough as a submit ensemble component yet.")


if __name__ == "__main__":
    main()
