import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import List

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
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


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append({"raw": line})
    return rows


def load_labels_csv(path: Path) -> np.ndarray:
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

    def norm(value):
        s = str(value).strip()
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

    print(f"Label column: {best_col}")
    return np.asarray(best_vals, dtype=np.int64)


def compact(obj, max_chars=8000):
    if obj is None:
        return ""
    if isinstance(obj, str):
        text = obj
    else:
        try:
            text = json.dumps(obj, ensure_ascii=False)
        except Exception:
            text = str(obj)
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[:max_chars] + " ..."
    return text


def flatten_action(value):
    if not isinstance(value, dict):
        return compact(value)
    parts = []
    name = value.get("name")
    if name:
        parts.append(f"assistant_action_name={name}")
    args = value.get("args")
    if args:
        parts.append(f"args={compact(args, 1000)}")
    result_summary = value.get("result_summary")
    if result_summary:
        parts.append(f"result={compact(result_summary, 1000)}")
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
    current = extract_current_prompt(row)
    if current:
        parts.append(f"CURRENT: {current}")

    history = row.get("history")
    if isinstance(history, list):
        for item in history[-8:]:
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
    elif history:
        parts.append(f"HISTORY: {compact(history)}")

    meta = row.get("session_meta")
    if isinstance(meta, dict):
        workspace = meta.get("workspace", {})
        if isinstance(workspace, dict):
            parts.append(f"workspace={compact(workspace, 800)}")
        language_pref = meta.get("language_pref")
        if language_pref:
            parts.append(f"language_pref={language_pref}")

    if not parts:
        parts.append(compact(row))
    return "\n".join(parts)


def extract_current_only_text(row):
    current = extract_current_prompt(row)
    return current if current else extract_text(row)


def get_last_action(row):
    if not isinstance(row, dict):
        return ""
    history = row.get("history")
    if not isinstance(history, list):
        return ""
    for item in reversed(history):
        if isinstance(item, dict) and item.get("role") == "assistant_action":
            return str(item.get("name", ""))
    return ""


def count_regex(patterns, text):
    if not text:
        return 0
    return sum(1 for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE))


def structured_tree_features(rows, texts, current_texts):
    features = []
    last_actions = []
    path_re = re.compile(r"[\w./\\-]+\.(py|ts|tsx|js|jsx|rs|go|java|kt|yaml|yml|json|toml|sh|md|txt|tf|sql|vue|tsx|css)", re.I)
    dir_re = re.compile(r"(src|app|lib|config|configs|scripts|tests|components|routes|models|data|dags|terraform|k8s|\.github|ios|android)(/|\\|\s|$)", re.I)
    cmd_re = re.compile(r"(npm|yarn|pnpm|pytest|python|go|cargo|docker|mvn|gradlew|ruff|tsc|vitest|jest|uvicorn|bash)\s+", re.I)

    for row, text, current in zip(rows, texts, current_texts):
        lower_text = text.lower()
        lower_current = current.lower()
        last_action = get_last_action(row)
        last_actions.append(last_action)

        f = [
            len(text),
            len(current),
            text.count("\n"),
            current.count("?") + current.count("？"),
            current.count("!") + current.count("！"),
            len(path_re.findall(text)),
            len(path_re.findall(current)),
            len(dir_re.findall(text)),
            len(dir_re.findall(current)),
            len(cmd_re.findall(text)),
            len(cmd_re.findall(current)),
            int("?" in current),
            int("ㅠ" in current or "ㅜ" in current),
            int("..." in current or "…" in current),
            int("ERROR" in text or "error" in text),
            int("PASS" in text or "passed" in lower_text or "green" in lower_text),
            int("FAIL" in text or "failed" in lower_text),
            int("permission denied" in lower_text),
            int("target_symbol" in lower_text),
            int("result_summary" in lower_text),
        ]

        for class_name in ALL_CLASSES:
            patterns = CUE_GROUPS.get(class_name, [])
            current_count = count_regex(patterns, current)
            text_count = count_regex(patterns, text)
            f.extend([current_count, text_count, int(current_count > 0), int(text_count > 0)])

        f.extend([
            count_regex(CUE_GROUPS["list_directory"], current) - count_regex(CUE_GROUPS["read_file"], current),
            count_regex(CUE_GROUPS["grep_search"], current) - count_regex(CUE_GROUPS["read_file"], current),
            count_regex(CUE_GROUPS["ask_user"], current) - count_regex(CUE_GROUPS["plan_task"], current),
            count_regex(CUE_GROUPS["lint_or_typecheck"], current) - count_regex(CUE_GROUPS["run_bash"], current),
            count_regex(CUE_GROUPS["web_search"], current) - count_regex(CUE_GROUPS["grep_search"], current),
        ])
        features.append(f)

    dense = np.asarray(features, dtype=np.float32)
    last_onehot = np.zeros((len(rows), len(ALL_CLASSES) + 1), dtype=np.float32)
    action_to_id = {name: i for i, name in enumerate(ALL_CLASSES)}
    for i, action in enumerate(last_actions):
        last_onehot[i, action_to_id.get(action, len(ALL_CLASSES))] = 1.0
    return np.hstack([dense, last_onehot])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/v23_tree_lgbm_full"))
    parser.add_argument("--word-features", type=int, default=30000)
    parser.add_argument("--char-features", type=int, default=30000)
    parser.add_argument("--word-ngram-max", type=int, default=3)
    parser.add_argument("--char-ngram-min", type=int, default=3)
    parser.add_argument("--char-ngram-max", type=int, default=5)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=450)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.65)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.5)
    parser.add_argument("--blend-weight", type=float, default=0.25)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--joblib-compress", type=int, default=3)
    args = parser.parse_args()

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        raise RuntimeError("lightgbm is required. Install with: python -m pip install lightgbm") from exc

    print("Load data...")
    rows = load_jsonl(args.data)
    y = load_labels_csv(args.labels_csv)
    if len(rows) != len(y):
        raise RuntimeError(f"data/label row mismatch: {len(rows)} vs {len(y)}")

    print("Extract text...")
    texts = [extract_text(row) for row in rows]
    current_texts = [extract_current_only_text(row) for row in rows]

    print("Vectorize TF-IDF...")
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        ngram_range=(1, args.word_ngram_max),
        max_features=args.word_features,
        min_df=args.min_df,
        sublinear_tf=True,
        strip_accents=None,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=(args.char_ngram_min, args.char_ngram_max),
        max_features=args.char_features,
        min_df=args.min_df,
        sublinear_tf=True,
    )

    X_word = word_vectorizer.fit_transform(texts)
    X_char = char_vectorizer.fit_transform(current_texts)

    print("Build structured cue features...")
    structured_dense = structured_tree_features(rows, texts, current_texts)
    scaler = StandardScaler()
    X_structured = sparse.csr_matrix(scaler.fit_transform(structured_dense))
    X = sparse.hstack([X_word, X_char, X_structured], format="csr")

    print(f"Feature matrix: {X.shape}, nnz={X.nnz}")
    print("Train LightGBM full tree model...")
    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(ALL_CLASSES),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=-1,
    )
    model.fit(X, y)

    bundle = {
        "format": "v23_tree_tfidf_structured_lgbm_v1",
        "classes": ALL_CLASSES,
        "model": model,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "scaler": scaler,
        "config": {
            "blend_weight": float(args.blend_weight),
            "word_features": int(args.word_features),
            "char_features": int(args.char_features),
            "word_ngram_max": int(args.word_ngram_max),
            "char_ngram_min": int(args.char_ngram_min),
            "char_ngram_max": int(args.char_ngram_max),
            "min_df": int(args.min_df),
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "max_depth": int(args.max_depth),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "reg_alpha": float(args.reg_alpha),
            "reg_lambda": float(args.reg_lambda),
            "seed": int(args.seed),
        },
    }

    artifact_path = args.output_dir / "tree_artifacts.joblib"
    print(f"Save tree artifacts: {artifact_path}")
    joblib.dump(bundle, artifact_path, compress=args.joblib_compress)

    config_path = args.output_dir / "tree_blend_config.json"
    config_path.write_text(
        json.dumps({"blend_weight": float(args.blend_weight), "artifact": "tree_artifacts.joblib"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    size_mb = artifact_path.stat().st_size / 1_000_000
    elapsed = time.perf_counter() - started
    print(f"Artifact size: {size_mb:.2f} MB")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print("Saved:")
    print(f"  {artifact_path}")
    print(f"  {config_path}")


if __name__ == "__main__":
    main()
