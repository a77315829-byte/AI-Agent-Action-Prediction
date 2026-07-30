import json
import math
import re
from typing import Any, Dict, List, Tuple

import numpy as np


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

FAMILY_NAMES = [
    "explore",
    "modify",
    "execute",
    "dialog",
    "external",
]

ACTION_TO_FAMILY = [
    0, 0, 0, 0,      # explore
    1, 1, 1,         # modify
    2, 2, 2,         # execute
    3, 3,            # ask_user, plan_task
    4,               # web_search
    3,               # respond_only
]

NONE_ACTION = "none"
ACTION_CATEGORIES = ALL_CLASSES + [NONE_ACTION]
CI_CATEGORIES = ["passed", "failed", "none", "unknown"]
TIER_CATEGORIES = ["enterprise", "pro", "free", "unknown"]
LANG_PREF_CATEGORIES = ["ko", "en", "mixed", "unknown"]
BOOL_CATEGORIES = ["true", "false", "unknown"]
SOURCE_CATEGORIES = ["sim", "au", "other"]
TOP_LANGUAGE_CATEGORIES = [
    "py", "js", "ts", "tsx", "jsx",
    "java", "go", "rs", "cpp", "c", "cs",
    "sql", "html", "css", "json", "yaml",
    "md", "other", "none",
]

FILE_PATH_PATTERN = re.compile(
    r"[\w.@+~:/\\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cpp|cc|c|h|hpp|"
    r"cs|css|html|json|ya?ml|md|xml|toml|ini|env|gradle|kt|swift|php|"
    r"rb|sql|sh|ps1|bat|vue|svelte)\b",
    re.IGNORECASE,
)

GLOB_PATTERN = re.compile(
    r"(?:\*\.[a-z0-9]+|\*\*/|\?[a-z0-9._-]+|\[[^\]]+\])",
    re.IGNORECASE,
)

FLAG_PATTERNS = {
    "has_open_intent": r"열어|읽어|내용|보여|확인해|살펴|훑어|open|read|show|inspect",
    "has_search_intent": r"찾아|검색|어디서|사용처|호출처|참조|정의|locate|find|search|grep|reference|usage",
    "has_directory_intent": r"디렉터리|폴더|구조|목록|나열|뭐뭐|directory|folder|tree|list",
    "has_all_files_intent": r"모든 파일|전체 파일|파일들|전부|싹|all files|every file|files matching",
    "has_test_intent": r"테스트|회귀|검증 테스트|pytest|jest|vitest|unittest|test suite|spec\b|go test|cargo test",
    "has_lint_intent": r"린트|타입체크|타입 검사|정적 분석|문법 검사|eslint|mypy|pyright|tsc|typecheck|lint|go vet|cargo check",
    "has_bash_intent": r"설치|실행해|명령어|터미널|쉘|npm install|pip install|git |docker|migrate|shell|bash|powershell",
    "has_web_intent": r"공식 문서|최신|권장|버전 확인|인터넷|웹 검색|documentation|official|latest|recommended|release note",
    "has_plan_intent": r"계획|단계|순서|접근법|작업 나눠|정리해|plan|steps|approach|break down|roadmap",
    "has_question_intent": r"어떤 걸|무엇을|뭘 선택|확인해줘\?|물어봐|추가 정보|which one|clarify|need more information",
    "has_multi_file_intent": r"두 파일|여러 파일|양쪽|모두 수정|한꺼번에|동시에|일괄|공통 부분|multiple files|both files|across files",
    "has_patch_intent": r"패치|diff|변경사항 적용|apply patch|patch",
    "has_create_intent": r"새 파일|파일 생성|만들어|추가해|create file|new file|write a file",
    "has_edit_intent": r"수정|바꿔|고쳐|변경|리팩터|edit|modify|change|fix|refactor",
    "has_run_intent": r"돌려|실행|확인해봐|run|execute",
    "has_error_intent": r"오류|에러|실패|예외|버그|error|failed|exception|bug|traceback",
}


def _safe_text(value: Any, limit: int = 1200) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)

    return " ".join(text.split())[:limit]


def _normalize_category(value: Any, allowed: List[str]) -> str:
    text = str(value).strip().lower()

    if text in allowed:
        return text

    return "unknown" if "unknown" in allowed else allowed[-1]


def _one_hot(value: str, categories: List[str]) -> List[float]:
    return [1.0 if value == category else 0.0 for category in categories]


def _source_from_id(sample_id: str) -> str:
    if sample_id.startswith("sess_sim_"):
        return "sim"

    if sample_id.startswith("sess_au_"):
        return "au"

    return "other"


def _normalize_language(language: Any) -> str:
    text = str(language).strip().lower()

    aliases = {
        "python": "py",
        "javascript": "js",
        "typescript": "ts",
        "rust": "rs",
        "c++": "cpp",
        "csharp": "cs",
        "c#": "cs",
        "yml": "yaml",
        "markdown": "md",
    }

    text = aliases.get(text, text)

    if text in TOP_LANGUAGE_CATEGORIES:
        return text

    return "other"


def _log_normalize(value: Any, maximum: float) -> float:
    try:
        number = max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0

    return min(1.0, math.log1p(number) / math.log1p(maximum))


def _bool_category(value: Any) -> str:
    if value is True or str(value).lower() == "true":
        return "true"

    if value is False or str(value).lower() == "false":
        return "false"

    return "unknown"


def extract_state(sample: Dict[str, Any]) -> Dict[str, Any]:
    history = sample.get("history", [])

    if not isinstance(history, list):
        history = []

    user_turns: List[str] = []
    action_items: List[Dict[str, str]] = []

    for item in history:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", ""))
        name = _safe_text(item.get("name", ""), 80)

        if role == "user":
            user_turns.append(
                _safe_text(item.get("content", ""), 700)
            )
        elif role == "assistant_action" or name:
            action_items.append({
                "name": name or NONE_ACTION,
                "args": _safe_text(item.get("args", ""), 500),
                "result": _safe_text(
                    item.get("result_summary", ""),
                    700,
                ),
            })

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

    if language_mix:
        top_language = _normalize_language(
            max(
                language_mix.items(),
                key=lambda item: item[1],
            )[0]
        )
    else:
        top_language = "none"

    current_prompt = _safe_text(
        sample.get("current_prompt", ""),
        1800,
    )

    previous_actions = [
        item["name"] for item in action_items
    ]

    failed_results = sum(
        bool(
            re.search(
                FLAG_PATTERNS["has_error_intent"],
                item["result"].lower(),
            )
        )
        for item in action_items
    )

    paths = FILE_PATH_PATTERN.findall(current_prompt)
    unique_paths = sorted(set(paths))

    return {
        "sample_id": str(sample.get("id", "")),
        "current_prompt": current_prompt,
        "user_turns": user_turns,
        "action_items": action_items,
        "previous_actions": previous_actions,
        "last_action": (
            previous_actions[-1]
            if previous_actions else NONE_ACTION
        ),
        "second_last_action": (
            previous_actions[-2]
            if len(previous_actions) >= 2
            else NONE_ACTION
        ),
        "history_len": len(history),
        "action_count": len(action_items),
        "failed_result_count": failed_results,
        "meta": meta,
        "workspace": workspace,
        "open_files": [str(path) for path in open_files],
        "language_mix": language_mix,
        "top_language": top_language,
        "unique_paths": unique_paths,
    }


def build_segments(sample: Dict[str, Any]) -> Dict[str, str]:
    state = extract_state(sample)

    recent_users = state["user_turns"][-4:]
    recent_actions = state["action_items"][-4:]

    history_text = "\n".join(
        f"[HISTORY_USER] {content}"
        for content in recent_users
    ) or "[HISTORY_USER] none"

    action_parts: List[str] = []

    for item in recent_actions:
        action_parts.extend([
            f"[HISTORY_ACTION] {item['name']}",
            f"[ACTION_ARGS] {item['args']}",
            f"[ACTION_RESULT] {item['result']}",
        ])

    action_parts.extend([
        (
            "[ACTION_SEQUENCE] "
            + (
                " > ".join(state["previous_actions"][-5:])
                if state["previous_actions"]
                else "none"
            )
        ),
        f"[LAST_ACTION] {state['last_action']}",
        f"[SECOND_LAST_ACTION] {state['second_last_action']}",
    ])

    action_text = "\n".join(action_parts)

    meta = state["meta"]
    workspace = state["workspace"]

    language_mix_text = " ".join(
        f"{key}={value}"
        for key, value in sorted(
            state["language_mix"].items()
        )
    ) or "none"

    meta_text = "\n".join([
        (
            "[SESSION] "
            f"tier={_safe_text(meta.get('user_tier'), 30)} "
            f"language={_safe_text(meta.get('language_pref'), 30)} "
            f"turn={_safe_text(meta.get('turn_index'), 30)} "
            f"budget={_safe_text(meta.get('budget_tokens_remaining'), 30)} "
            f"elapsed={_safe_text(meta.get('elapsed_session_sec'), 30)}"
        ),
        (
            "[WORKSPACE] "
            f"git_dirty={_safe_text(workspace.get('git_dirty'), 20)} "
            f"ci={_safe_text(workspace.get('last_ci_status'), 30)} "
            f"loc={_safe_text(workspace.get('loc'), 30)} "
            f"top_language={state['top_language']}"
        ),
        (
            "[OPEN_FILES] "
            + (
                " | ".join(state["open_files"][-6:])
                if state["open_files"]
                else "none"
            )
        ),
        f"[LANGUAGE_MIX] {language_mix_text}",
    ])

    current_text = "\n".join([
        f"[CURRENT_PROMPT] {state['current_prompt']}",
        "[TASK] Predict the next coding-agent action.",
    ])

    return {
        "history": history_text,
        "action": action_text,
        "meta": meta_text,
        "current": current_text,
    }


def structured_feature_names() -> List[str]:
    names: List[str] = []

    names.extend(
        f"last_action={value}"
        for value in ACTION_CATEGORIES
    )
    names.extend(
        f"second_last_action={value}"
        for value in ACTION_CATEGORIES
    )
    names.extend(
        f"ci={value}" for value in CI_CATEGORIES
    )
    names.extend(
        f"tier={value}" for value in TIER_CATEGORIES
    )
    names.extend(
        f"language_pref={value}"
        for value in LANG_PREF_CATEGORIES
    )
    names.extend(
        f"git_dirty={value}"
        for value in BOOL_CATEGORIES
    )
    names.extend(
        f"source={value}"
        for value in SOURCE_CATEGORIES
    )
    names.extend(
        f"top_language={value}"
        for value in TOP_LANGUAGE_CATEGORIES
    )

    names.extend([
        "turn_index_norm",
        "history_len_norm",
        "action_count_norm",
        "open_file_count_norm",
        "budget_norm",
        "workspace_loc_norm",
        "elapsed_norm",
        "prompt_length_norm",
        "file_path_count_norm",
        "failed_result_count_norm",
    ])

    names.extend([
        "has_exact_file_path",
        "has_glob_expression",
        *FLAG_PATTERNS.keys(),
    ])

    return names


STRUCTURED_FEATURE_NAMES = structured_feature_names()
STRUCTURED_DIM = len(STRUCTURED_FEATURE_NAMES)


def build_structured_features(
    sample: Dict[str, Any],
) -> np.ndarray:
    state = extract_state(sample)
    meta = state["meta"]
    workspace = state["workspace"]
    prompt = state["current_prompt"]
    prompt_lower = prompt.lower()

    last_action = (
        state["last_action"]
        if state["last_action"] in ACTION_CATEGORIES
        else NONE_ACTION
    )
    second_last_action = (
        state["second_last_action"]
        if state["second_last_action"] in ACTION_CATEGORIES
        else NONE_ACTION
    )

    ci_status = _normalize_category(
        workspace.get("last_ci_status", "unknown"),
        CI_CATEGORIES,
    )
    user_tier = _normalize_category(
        meta.get("user_tier", "unknown"),
        TIER_CATEGORIES,
    )
    language_pref = _normalize_category(
        meta.get("language_pref", "unknown"),
        LANG_PREF_CATEGORIES,
    )
    git_dirty = _bool_category(
        workspace.get("git_dirty")
    )
    source = _source_from_id(state["sample_id"])

    features: List[float] = []

    features.extend(
        _one_hot(last_action, ACTION_CATEGORIES)
    )
    features.extend(
        _one_hot(
            second_last_action,
            ACTION_CATEGORIES,
        )
    )
    features.extend(
        _one_hot(ci_status, CI_CATEGORIES)
    )
    features.extend(
        _one_hot(user_tier, TIER_CATEGORIES)
    )
    features.extend(
        _one_hot(
            language_pref,
            LANG_PREF_CATEGORIES,
        )
    )
    features.extend(
        _one_hot(git_dirty, BOOL_CATEGORIES)
    )
    features.extend(
        _one_hot(source, SOURCE_CATEGORIES)
    )
    features.extend(
        _one_hot(
            state["top_language"],
            TOP_LANGUAGE_CATEGORIES,
        )
    )

    features.extend([
        _log_normalize(meta.get("turn_index"), 40),
        min(1.0, state["history_len"] / 12.0),
        min(1.0, state["action_count"] / 8.0),
        min(1.0, len(state["open_files"]) / 10.0),
        _log_normalize(
            meta.get("budget_tokens_remaining"),
            250_000,
        ),
        _log_normalize(workspace.get("loc"), 2_000_000),
        _log_normalize(
            meta.get("elapsed_session_sec"),
            14_400,
        ),
        min(1.0, len(prompt) / 2000.0),
        min(1.0, len(state["unique_paths"]) / 6.0),
        min(
            1.0,
            state["failed_result_count"] / 5.0,
        ),
    ])

    features.append(
        1.0 if state["unique_paths"] else 0.0
    )
    features.append(
        1.0 if GLOB_PATTERN.search(prompt) else 0.0
    )

    for pattern in FLAG_PATTERNS.values():
        features.append(
            1.0 if re.search(pattern, prompt_lower) else 0.0
        )

    array = np.asarray(
        features,
        dtype=np.float32,
    )

    if array.shape[0] != STRUCTURED_DIM:
        raise RuntimeError(
            f"structured feature dim mismatch: "
            f"{array.shape[0]} != {STRUCTURED_DIM}"
        )

    return array


def family_id_from_action_id(action_id: int) -> int:
    return ACTION_TO_FAMILY[int(action_id)]


# ============================================================
# V13 clean extra structured features
# ============================================================
# IMPORTANT:
# - The original V4 STRUCTURED_DIM and build_structured_features() are kept
#   unchanged so V4/V12 heads can be loaded without reshaping any existing
#   LayerNorm/Linear weights.
# - These extra features must be consumed by a separate residual branch.

RECENT_ACTION_SLOTS = 8

OPEN_FILE_EXT_CATEGORIES = [
    "py", "js", "ts", "tsx", "jsx", "java", "go", "rs", "cpp", "c", "cs",
    "sql", "html", "css", "json", "yaml", "md", "sh", "toml", "ini", "env",
    "xml", "other",
]

LANGUAGE_RATIO_CATEGORIES = [
    "py", "js", "ts", "tsx", "jsx", "java", "go", "rs", "cpp", "c", "cs",
    "sql", "html", "css", "json", "yaml", "md", "other",
]

WORKFLOW_PATTERNS = {
    "workflow_test_only": (
        r"(테스트|pytest|jest|vitest|unittest|go test|cargo test|run tests?)"
        r"(?!.*(수정|고쳐|fix|edit|patch|change|apply))"
    ),
    "workflow_run_then_fix": r"(실행|돌려|run|execute).*(실패|오류|에러|고쳐|수정|fix|failed|error)",
    "workflow_fix_then_validate": r"(고쳐|수정|fix|edit|patch).*(테스트|검증|확인|lint|typecheck|pytest|jest|run)",
    "workflow_search_then_edit": r"(찾아|검색|grep|search|locate|reference|usage).*(수정|고쳐|edit|fix|patch)",
    "workflow_read_then_edit": r"(읽어|열어|확인|read|open|inspect).*(수정|고쳐|edit|fix|patch)",
    "workflow_error_then_search": r"(오류|에러|traceback|exception|failed|bug).*(찾아|검색|search|grep|why|원인)",
    "workflow_lint_only": r"(린트|타입체크|eslint|mypy|pyright|tsc|lint|typecheck)(?!.*(수정|고쳐|fix|edit|patch))",
    "workflow_lint_then_fix": r"(린트|타입체크|eslint|mypy|pyright|tsc|lint|typecheck).*(수정|고쳐|fix|edit|patch)",
    "workflow_ask_clarification": r"(물어봐|질문|확인 질문|clarify|ask|need more information|which one)",
    "workflow_web_needed": r"(최신|공식 문서|인터넷|웹|버전 확인|latest|official|documentation|release note)",
    "workflow_create_new_file": r"(새 파일|파일 생성|만들어|추가해|create file|new file|write a file)",
    "workflow_multi_file_change": r"(여러 파일|두 파일|양쪽|모두 수정|일괄|multiple files|both files|across files)",
    "workflow_patch_requested": r"(패치|diff|apply patch|변경사항 적용)",
    "workflow_explain_only": r"(설명|왜|원리|정리|요약|explain|summarize|why|what is)",
}

TOOL_RESULT_PATTERNS = {
    "last_result_has_error": r"오류|에러|error|exception|traceback|failed|failure|fatal",
    "last_result_has_success": r"success|passed|완료|성공|saved|created|done|ok\b",
    "last_result_has_test": r"pytest|jest|vitest|unittest|test|tests|spec",
    "last_result_has_lint": r"eslint|mypy|pyright|tsc|lint|typecheck|type check",
    "last_result_has_not_found": r"not found|no such file|없습니다|찾을 수|cannot find|missing",
    "last_result_has_timeout": r"timeout|timed out|시간 초과",
    "last_result_has_permission": r"permission|denied|권한",
    "last_result_has_empty": r"no output|empty|0 results|결과 없음",
    "last_result_has_warning": r"warning|warn|경고",
    "last_result_has_install": r"install|installed|pip|npm|package|dependency",
}

PROJECT_FILE_NAMES = {
    "package.json", "pyproject.toml", "setup.py", "requirements.txt",
    "tsconfig.json", "vite.config.ts", "next.config.js", "dockerfile",
    "docker-compose.yml", "compose.yml", "eslint.config.js", ".eslintrc.json",
}


def _extension_from_path(path: str) -> str:
    text = str(path).replace("\\", "/").lower()
    name = text.rsplit("/", 1)[-1]
    if name in {"dockerfile", "makefile"}:
        return "other"
    if name.endswith(".env") or ".env." in name:
        return "env"
    if "." not in name:
        return "other"
    ext = name.rsplit(".", 1)[-1]
    aliases = {
        "yml": "yaml",
        "markdown": "md",
        "bash": "sh",
        "zsh": "sh",
        "ps1": "sh",
        "bat": "sh",
        "cmd": "sh",
        "cxx": "cpp",
        "cc": "cpp",
        "hpp": "cpp",
        "h": "c",
    }
    ext = aliases.get(ext, ext)
    if ext in OPEN_FILE_EXT_CATEGORIES:
        return ext
    return "other"


def _path_depth(path: str) -> int:
    text = str(path).replace("\\", "/").strip("/")
    if not text:
        return 0
    return max(0, text.count("/"))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _normalized_entropy(counts: List[float]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    probs = [value / total for value in counts if value > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return float(entropy / math.log(len(counts)))


def extra_structured_feature_names() -> List[str]:
    names: List[str] = []

    for slot in range(RECENT_ACTION_SLOTS):
        names.extend(
            f"recent_action_slot_{slot + 1}={value}"
            for value in ACTION_CATEGORIES
        )

    names.extend(
        f"recent_action_count={value}"
        for value in ALL_CLASSES
    )
    names.extend(
        f"recent_family_count={value}"
        for value in FAMILY_NAMES
    )
    for previous_family in FAMILY_NAMES:
        for next_family in FAMILY_NAMES:
            names.append(
                f"recent_family_transition={previous_family}->{next_family}"
            )

    names.extend([
        "recent_action_unique_norm",
        "recent_action_entropy_norm",
        "recent_action_streak_norm",
        "last_equals_second_action",
        "last_family_changed",
    ])

    names.extend(
        f"open_ext_ratio={value}"
        for value in OPEN_FILE_EXT_CATEGORIES
    )
    names.extend([
        "open_file_count_v13_norm",
        "open_path_mean_depth_norm",
        "open_path_max_depth_norm",
        "open_has_test_file",
        "open_has_config_file",
        "open_has_source_file",
        "open_has_doc_file",
        "open_has_lock_file",
        "open_has_env_file",
        "open_has_project_file",
        "open_has_package_json",
        "open_has_pyproject",
        "open_has_tsconfig",
        "open_has_requirements",
        "open_has_docker",
        "prompt_filename_overlap_norm",
        "prompt_extension_overlap_norm",
        "prompt_mentions_open_path",
    ])

    names.extend(
        f"language_ratio={value}"
        for value in LANGUAGE_RATIO_CATEGORIES
    )
    names.extend([
        "language_mix_total_norm",
        "language_mix_diversity_norm",
        "stack_python_hint",
        "stack_node_hint",
        "stack_typescript_hint",
        "stack_frontend_hint",
        "stack_backend_hint",
        "stack_test_hint",
        "stack_lint_hint",
        "stack_docker_hint",
        "stack_database_hint",
        "stack_config_hint",
        "stack_docs_hint",
    ])

    names.extend(WORKFLOW_PATTERNS.keys())
    names.extend(TOOL_RESULT_PATTERNS.keys())
    names.extend([
        "recent_result_error_count_norm",
        "recent_result_success_count_norm",
        "recent_result_empty_count_norm",
    ])

    return names


EXTRA_STRUCTURED_FEATURE_NAMES = extra_structured_feature_names()
EXTRA_STRUCTURED_DIM = len(EXTRA_STRUCTURED_FEATURE_NAMES)


def build_extra_structured_features(sample: Dict[str, Any]) -> np.ndarray:
    state = extract_state(sample)
    prompt = state["current_prompt"]
    prompt_lower = prompt.lower()
    features: List[float] = []

    recent_actions = list(state["previous_actions"][-RECENT_ACTION_SLOTS:])
    padded_actions = [NONE_ACTION] * (RECENT_ACTION_SLOTS - len(recent_actions)) + recent_actions

    for action_name in padded_actions:
        normalized_action = action_name if action_name in ACTION_CATEGORIES else NONE_ACTION
        features.extend(_one_hot(normalized_action, ACTION_CATEGORIES))

    recent_non_none = [
        action for action in recent_actions
        if action in ALL_CLASSES
    ]
    action_counts = {label: 0 for label in ALL_CLASSES}
    for action in recent_non_none:
        action_counts[action] += 1
    denominator = float(max(1, RECENT_ACTION_SLOTS))
    features.extend(action_counts[label] / denominator for label in ALL_CLASSES)

    family_counts = [0.0 for _ in FAMILY_NAMES]
    family_sequence: List[int] = []
    for action in recent_non_none:
        action_id = ALL_CLASSES.index(action)
        family_id = int(ACTION_TO_FAMILY[action_id])
        family_counts[family_id] += 1.0
        family_sequence.append(family_id)
    features.extend(value / denominator for value in family_counts)

    transition_counts = np.zeros((len(FAMILY_NAMES), len(FAMILY_NAMES)), dtype=np.float32)
    for previous, current in zip(family_sequence[:-1], family_sequence[1:]):
        transition_counts[previous, current] += 1.0
    transition_denominator = float(max(1, len(family_sequence) - 1))
    features.extend((transition_counts / transition_denominator).reshape(-1).tolist())

    if recent_non_none:
        last = recent_non_none[-1]
        streak = 1
        for action in reversed(recent_non_none[:-1]):
            if action == last:
                streak += 1
            else:
                break
    else:
        streak = 0
    features.extend([
        min(1.0, len(set(recent_non_none)) / float(len(ALL_CLASSES))),
        _normalized_entropy([float(action_counts[label]) for label in ALL_CLASSES]),
        min(1.0, streak / float(RECENT_ACTION_SLOTS)),
        1.0 if len(recent_non_none) >= 2 and recent_non_none[-1] == recent_non_none[-2] else 0.0,
        1.0 if len(family_sequence) >= 2 and family_sequence[-1] != family_sequence[-2] else 0.0,
    ])

    open_files = [str(path) for path in state["open_files"]]
    open_count = len(open_files)
    open_denominator = float(max(1, open_count))
    ext_counts = {ext: 0 for ext in OPEN_FILE_EXT_CATEGORIES}
    basenames: List[str] = []
    extensions: List[str] = []
    depths: List[int] = []
    lowered_files: List[str] = []

    for path in open_files:
        lowered = path.replace("\\", "/").lower()
        lowered_files.append(lowered)
        basename = lowered.rsplit("/", 1)[-1]
        basenames.append(basename)
        ext = _extension_from_path(path)
        extensions.append(ext)
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        depths.append(_path_depth(path))

    features.extend(ext_counts[ext] / open_denominator for ext in OPEN_FILE_EXT_CATEGORIES)

    mean_depth = _safe_ratio(sum(depths), len(depths)) if depths else 0.0
    max_depth = max(depths) if depths else 0
    open_text = " ".join(lowered_files)
    prompt_tokens = set(re.findall(r"[a-zA-Z0-9_@+.-]+", prompt_lower))
    filename_overlap = sum(1 for name in basenames if name and name in prompt_lower)
    extension_overlap = sum(1 for ext in set(extensions) if ext != "other" and re.search(rf"\b{re.escape(ext)}\b", prompt_lower))
    prompt_mentions_path = any(path and path in prompt_lower for path in lowered_files)

    test_file_pattern = r"(^|[/\\])(tests?|__tests__|spec)([/\\]|$)|(_test|\.test|\.spec)\."
    config_pattern = r"config|\.json$|\.ya?ml$|\.toml$|\.ini$|tsconfig|eslint|prettier|package\.json|pyproject"
    source_pattern = r"\.(py|js|ts|tsx|jsx|java|go|rs|cpp|cc|c|cs|sql|html|css)$"
    doc_pattern = r"\.(md|rst|txt)$|readme|docs?/"

    features.extend([
        min(1.0, open_count / 12.0),
        min(1.0, mean_depth / 8.0),
        min(1.0, max_depth / 12.0),
        1.0 if re.search(test_file_pattern, open_text) else 0.0,
        1.0 if re.search(config_pattern, open_text) else 0.0,
        1.0 if re.search(source_pattern, open_text) else 0.0,
        1.0 if re.search(doc_pattern, open_text) else 0.0,
        1.0 if re.search(r"lock$|package-lock\.json|yarn\.lock|pnpm-lock|poetry\.lock", open_text) else 0.0,
        1.0 if re.search(r"(^|[/\\])\.env|\.env\.", open_text) else 0.0,
        1.0 if any(name in PROJECT_FILE_NAMES for name in basenames) else 0.0,
        1.0 if "package.json" in basenames else 0.0,
        1.0 if "pyproject.toml" in basenames else 0.0,
        1.0 if "tsconfig.json" in basenames else 0.0,
        1.0 if "requirements.txt" in basenames else 0.0,
        1.0 if "dockerfile" in basenames or "docker-compose.yml" in basenames else 0.0,
        min(1.0, filename_overlap / max(1.0, float(open_count))),
        min(1.0, extension_overlap / max(1.0, float(len(set(extensions))))),
        1.0 if prompt_mentions_path else 0.0,
    ])

    language_mix = state["language_mix"]
    normalized_language_counts = {category: 0.0 for category in LANGUAGE_RATIO_CATEGORIES}
    language_total = 0.0
    if isinstance(language_mix, dict):
        for key, value in language_mix.items():
            language = _normalize_language(key)
            if language not in normalized_language_counts:
                language = "other"
            try:
                amount = max(0.0, float(value))
            except (TypeError, ValueError):
                amount = 0.0
            normalized_language_counts[language] += amount
            language_total += amount
    language_denominator = max(1.0, language_total)
    features.extend(normalized_language_counts[language] / language_denominator for language in LANGUAGE_RATIO_CATEGORIES)
    features.extend([
        _log_normalize(language_total, 2_000_000),
        _normalized_entropy([normalized_language_counts[language] for language in LANGUAGE_RATIO_CATEGORIES]),
    ])

    combined_text = " ".join([prompt_lower, open_text])
    features.extend([
        1.0 if re.search(r"pytest|requirements|pyproject|\.py\b|python", combined_text) else 0.0,
        1.0 if re.search(r"package\.json|npm|yarn|pnpm|node_modules|node\b", combined_text) else 0.0,
        1.0 if re.search(r"typescript|tsconfig|\.ts\b|\.tsx\b|tsc\b", combined_text) else 0.0,
        1.0 if re.search(r"react|vue|svelte|next\.?js|vite|frontend|\.tsx\b|\.jsx\b", combined_text) else 0.0,
        1.0 if re.search(r"fastapi|django|flask|express|server|backend|api/|controller|route", combined_text) else 0.0,
        1.0 if re.search(r"pytest|jest|vitest|unittest|spec|test", combined_text) else 0.0,
        1.0 if re.search(r"eslint|mypy|pyright|ruff|black|prettier|tsc|lint|typecheck", combined_text) else 0.0,
        1.0 if re.search(r"docker|compose|container|dockerfile", combined_text) else 0.0,
        1.0 if re.search(r"sql|postgres|mysql|sqlite|mongodb|redis|database|migration|alembic", combined_text) else 0.0,
        1.0 if re.search(r"config|env|yaml|json|toml|ini|settings", combined_text) else 0.0,
        1.0 if re.search(r"readme|docs?|markdown|\.md\b|문서", combined_text) else 0.0,
    ])

    for pattern in WORKFLOW_PATTERNS.values():
        features.append(1.0 if re.search(pattern, prompt_lower) else 0.0)

    action_items = state["action_items"]
    last_result = action_items[-1]["result"].lower() if action_items else ""
    for pattern in TOOL_RESULT_PATTERNS.values():
        features.append(1.0 if re.search(pattern, last_result) else 0.0)

    recent_results = [item["result"].lower() for item in action_items[-RECENT_ACTION_SLOTS:]]
    features.extend([
        min(1.0, sum(1 for result in recent_results if re.search(TOOL_RESULT_PATTERNS["last_result_has_error"], result)) / float(RECENT_ACTION_SLOTS)),
        min(1.0, sum(1 for result in recent_results if re.search(TOOL_RESULT_PATTERNS["last_result_has_success"], result)) / float(RECENT_ACTION_SLOTS)),
        min(1.0, sum(1 for result in recent_results if re.search(TOOL_RESULT_PATTERNS["last_result_has_empty"], result)) / float(RECENT_ACTION_SLOTS)),
    ])

    array = np.asarray(features, dtype=np.float32)
    if array.shape[0] != EXTRA_STRUCTURED_DIM:
        raise RuntimeError(
            f"extra structured feature dim mismatch: {array.shape[0]} != {EXTRA_STRUCTURED_DIM}"
        )
    return array


# Alias kept short for training/inference code.
build_extra_features = build_extra_structured_features
