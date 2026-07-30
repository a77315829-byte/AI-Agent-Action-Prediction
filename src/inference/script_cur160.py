import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from scipy import sparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
QWEN_CONFIG_DIR = MODEL_DIR / "qwen_config"
QWEN_INT8_DIR = MODEL_DIR / "qwen_int8"
DATA_DIR = Path(os.environ.get("DACON_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("DACON_OUTPUT_DIR", ROOT / "output"))
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"
TREE_ARTIFACTS_PATH = MODEL_DIR / "tree_artifacts.joblib"
TREE_CONFIG_PATH = MODEL_DIR / "tree_blend_config.json"

sys.path.insert(0, str(MODEL_DIR))
from feature_utils_qwen_v4 import (  # noqa: E402
    ACTION_TO_FAMILY,
    ALL_CLASSES,
    FAMILY_NAMES,
    STRUCTURED_DIM,
    build_segments,
    build_structured_features,
)

MAX_LENGTH = 256
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))

SEGMENT_BUDGETS = {
    "history": 56,
    "action": 72,
    "meta": 32,
    "current": 96,
}
SEGMENT_IDS = {
    "history": 0,
    "action": 1,
    "meta": 2,
    "current": 3,
}


def load_jsonl(path: Path) -> List[dict]:
    samples: List[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def smart_trim(
    token_ids: List[int],
    budget: int,
    prefix_tokens: int = 8,
) -> List[int]:
    if len(token_ids) <= budget:
        return token_ids

    prefix_tokens = min(
        prefix_tokens,
        max(0, budget // 3),
    )
    if prefix_tokens == 0:
        return token_ids[-budget:]

    return (
        token_ids[:prefix_tokens]
        + token_ids[-(budget - prefix_tokens):]
    )


class EncodedDataset(Dataset):
    def __init__(
        self,
        samples: List[dict],
        tokenizer,
        chunk_size: int = 2048,
    ):
        self.input_ids: List[np.ndarray] = []
        self.segment_ids: List[np.ndarray] = []
        self.structured_features: List[np.ndarray] = []

        segments = [
            build_segments(sample)
            for sample in samples
        ]
        self.structured_features = [
            build_structured_features(sample)
            for sample in samples
        ]

        for start in tqdm(
            range(0, len(samples), chunk_size),
            desc="Tokenize test",
        ):
            batch_segments = segments[start:start + chunk_size]
            encoded_by_segment: Dict[str, List[List[int]]] = {}

            for segment_name in (
                "history",
                "action",
                "meta",
                "current",
            ):
                encoded_by_segment[segment_name] = tokenizer(
                    [
                        item[segment_name]
                        for item in batch_segments
                    ],
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                )["input_ids"]

            for local_index in range(len(batch_segments)):
                combined_ids: List[int] = []
                combined_segment_ids: List[int] = []

                for segment_name in (
                    "history",
                    "action",
                    "meta",
                    "current",
                ):
                    tokens = smart_trim(
                        encoded_by_segment[segment_name][local_index],
                        SEGMENT_BUDGETS[segment_name],
                    )
                    combined_ids.extend(tokens)
                    combined_segment_ids.extend(
                        [SEGMENT_IDS[segment_name]] * len(tokens)
                    )

                if not combined_ids:
                    fallback = (
                        tokenizer.eos_token_id
                        if tokenizer.eos_token_id is not None
                        else tokenizer.pad_token_id
                    )
                    combined_ids = [fallback]
                    combined_segment_ids = [SEGMENT_IDS["current"]]

                self.input_ids.append(
                    np.asarray(
                        combined_ids[-MAX_LENGTH:],
                        dtype=np.int32,
                    )
                )
                self.segment_ids.append(
                    np.asarray(
                        combined_segment_ids[-MAX_LENGTH:],
                        dtype=np.int8,
                    )
                )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict:
        return {
            "input_ids": self.input_ids[index],
            "segment_ids": self.segment_ids[index],
            "structured_features": self.structured_features[index],
        }


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: List[dict]) -> dict:
        batch_size = len(features)
        max_length = max(
            len(feature["input_ids"])
            for feature in features
        )
        padded_length = ((max_length + 7) // 8) * 8

        input_ids = torch.full(
            (batch_size, padded_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, padded_length),
            dtype=torch.long,
        )
        segment_ids = torch.full(
            (batch_size, padded_length),
            -1,
            dtype=torch.long,
        )
        structured_features = torch.tensor(
            np.stack([
                feature["structured_features"]
                for feature in features
            ]),
            dtype=torch.float32,
        )

        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.from_numpy(
                feature["input_ids"].astype(np.int64, copy=False)
            )
            attention_mask[row, :length] = 1
            segment_ids[row, :length] = torch.from_numpy(
                feature["segment_ids"].astype(np.int64, copy=False)
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "segment_ids": segment_ids,
            "structured_features": structured_features,
        }


class SegmentProjector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(
            self.linear(
                self.norm(values.float())
            )
        )


class QwenSegmentClassifier(nn.Module):
    def __init__(
        self,
        backbone,
        hidden_size: int,
        structured_dim: int,
        num_labels: int,
        num_families: int,
    ):
        super().__init__()
        self.backbone = backbone

        projection_size = 256
        self.current_projector = SegmentProjector(
            hidden_size,
            projection_size,
        )
        self.action_projector = SegmentProjector(
            hidden_size,
            projection_size,
        )
        self.history_projector = SegmentProjector(
            hidden_size,
            projection_size,
        )
        self.all_projector = SegmentProjector(
            hidden_size,
            projection_size,
        )

        self.structured_mlp = nn.Sequential(
            nn.LayerNorm(structured_dim),
            nn.Linear(structured_dim, 192),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(192, 128),
            nn.GELU(),
        )

        fusion_input = projection_size * 4 + 128
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input),
            nn.Linear(fusion_input, 512),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.action_head = nn.Linear(512, num_labels)
        self.family_head = nn.Linear(512, num_families)

    @staticmethod
    def masked_mean(
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_float = mask.unsqueeze(-1).to(hidden.dtype)
        total = (hidden * mask_float).sum(dim=1)
        denominator = mask_float.sum(dim=1)
        pooled = total / denominator.clamp(min=1.0)

        empty = denominator.squeeze(-1).eq(0)
        if empty.any():
            pooled[empty] = 0

        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        structured_features: torch.Tensor,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state

        current_pool = self.masked_mean(
            hidden,
            segment_ids.eq(SEGMENT_IDS["current"]),
        )
        action_pool = self.masked_mean(
            hidden,
            segment_ids.eq(SEGMENT_IDS["action"]),
        )
        history_pool = self.masked_mean(
            hidden,
            segment_ids.eq(SEGMENT_IDS["history"]),
        )
        all_pool = self.masked_mean(
            hidden,
            attention_mask.bool(),
        )

        fused = torch.cat([
            self.current_projector(current_pool),
            self.action_projector(action_pool),
            self.history_projector(history_pool),
            self.all_projector(all_pool),
            self.structured_mlp(structured_features),
        ], dim=-1)

        representation = self.fusion(fused)
        return (
            self.action_head(representation),
            self.family_head(representation),
        )


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


def aligned_tree_predict_proba(model, features) -> np.ndarray:
    local = model.predict_proba(features)
    result = np.full((features.shape[0], len(ALL_CLASSES)), 1e-12, dtype=np.float64)
    classes = getattr(model, "classes_", np.arange(local.shape[1]))
    result[:, classes.astype(int)] = local
    result /= result.sum(axis=1, keepdims=True)
    return result


def predict_tree_probabilities(tree_bundle: dict, samples: List[dict]) -> np.ndarray:
    print("Tree feature extraction...")
    texts = [extract_text(sample) for sample in samples]
    current_texts = [extract_current_only_text(sample) for sample in samples]

    word_vectorizer = tree_bundle["word_vectorizer"]
    char_vectorizer = tree_bundle["char_vectorizer"]
    scaler = tree_bundle["scaler"]
    model = tree_bundle["model"]

    x_word = word_vectorizer.transform(texts)
    x_char = char_vectorizer.transform(current_texts)
    structured_dense = structured_tree_features(samples, texts, current_texts)
    x_structured = sparse.csr_matrix(scaler.transform(structured_dense))
    features = sparse.hstack([x_word, x_char, x_structured], format="csr")

    print(f"Tree predict_proba features={features.shape}")
    return aligned_tree_predict_proba(model, features)


def safe_torch_load(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )




def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return torch.float16


def dequantize_tensor(entry: dict) -> torch.Tensor:
    kind = entry.get("kind")
    if kind == "raw":
        return entry["tensor"]

    q = entry["q"]
    scale = entry["scale"]
    target_dtype = _dtype_from_name(entry.get("target_dtype", "float16"))

    if not torch.is_tensor(scale):
        scale = torch.tensor(scale, dtype=torch.float32)

    # Dequantize directly to half precision to keep CPU memory below the
    # competition limit. Per-channel scales are stored with keepdim=True
    # broadcast metadata for 2D+ tensors.
    values = q.to(torch.float16)
    scale_values = scale.to(torch.float16)
    view_shape = entry.get("scale_view_shape")
    if view_shape:
        scale_values = scale_values.view(*view_shape)
    values = values * scale_values

    if target_dtype != torch.float16:
        values = values.to(target_dtype)
    return values


def load_int8_packed_qwen(
    config_dir: Path,
    int8_dir: Path,
    pad_token_id: int,
):
    index_path = int8_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))

    config = AutoConfig.from_pretrained(
        str(config_dir),
        local_files_only=True,
    )
    config.use_cache = False
    config.pad_token_id = pad_token_id

    print("Instantiate Qwen backbone from config...")
    original_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        backbone = AutoModel.from_config(
            config,
            attn_implementation="sdpa",
        )
    finally:
        torch.set_default_dtype(original_default_dtype)

    backbone.config.use_cache = False
    backbone.config.pad_token_id = pad_token_id

    state_dict = {}
    shards = index.get("shards", [])
    print(f"Load int8 Qwen shards: {len(shards)}")
    for shard_name in shards:
        shard_path = int8_dir / shard_name
        shard = safe_torch_load(shard_path)
        for name, entry in shard.items():
            state_dict[name] = dequantize_tensor(entry)
        del shard

    print("Load dequantized Qwen state dict...")
    backbone.load_state_dict(state_dict, strict=True)
    del state_dict
    return backbone

def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def build_calibrator_features(
    samples: List[dict],
    action_logits: np.ndarray,
    family_logits: np.ndarray,
    final_logits: np.ndarray,
) -> np.ndarray:
    probabilities = stable_softmax(final_logits)
    centered_logits = (
        final_logits
        - final_logits.mean(axis=1, keepdims=True)
    )

    sorted_probabilities = np.sort(probabilities, axis=1)
    margin = (
        sorted_probabilities[:, -1]
        - sorted_probabilities[:, -2]
    )[:, None]
    max_probability = probabilities.max(
        axis=1,
        keepdims=True,
    )
    entropy = (
        -np.sum(
            probabilities
            * np.log(np.maximum(probabilities, 1e-12)),
            axis=1,
        )
    )[:, None]

    structured = np.stack([
        build_structured_features(sample)
        for sample in samples
    ]).astype(np.float32)

    predicted_ids = final_logits.argmax(axis=1)
    predicted_one_hot = np.eye(
        len(ALL_CLASSES),
        dtype=np.float32,
    )[predicted_ids]

    source_sim = np.asarray([
        float(
            str(sample.get("id", "")).startswith("sess_sim_")
        )
        for sample in samples
    ], dtype=np.float32)[:, None]

    return np.hstack([
        action_logits.astype(np.float32),
        family_logits.astype(np.float32),
        final_logits.astype(np.float32),
        centered_logits.astype(np.float32),
        probabilities.astype(np.float32),
        predicted_one_hot,
        structured,
        margin.astype(np.float32),
        max_probability.astype(np.float32),
        entropy.astype(np.float32),
        source_sim,
    ]).astype(np.float32)


def aligned_predict_proba(
    model,
    features: np.ndarray,
) -> np.ndarray:
    local = model.predict_proba(features)

    result = np.full(
        (len(features), len(ALL_CLASSES)),
        1e-9,
        dtype=np.float64,
    )
    result[:, model.classes_.astype(int)] = local
    result /= result.sum(axis=1, keepdims=True)
    return result


def apply_calibrator(
    qwen_probabilities: np.ndarray,
    calibrator_probabilities: np.ndarray,
    config: dict,
) -> np.ndarray:
    alpha = float(config["alpha"])
    minimum_confidence = float(
        config["minimum_calibrator_confidence"]
    )
    maximum_margin = float(
        config["maximum_qwen_margin"]
    )

    qwen_prediction = qwen_probabilities.argmax(axis=1)
    calibrator_prediction = (
        calibrator_probabilities.argmax(axis=1)
    )

    sorted_qwen = np.sort(qwen_probabilities, axis=1)
    qwen_margin = (
        sorted_qwen[:, -1]
        - sorted_qwen[:, -2]
    )
    calibrator_confidence = (
        calibrator_probabilities.max(axis=1)
    )

    blended_log_probability = (
        (1.0 - alpha)
        * np.log(np.maximum(qwen_probabilities, 1e-12))
        + alpha
        * np.log(np.maximum(calibrator_probabilities, 1e-12))
    )
    blended_prediction = blended_log_probability.argmax(axis=1)

    use_calibrator = (
        (calibrator_confidence >= minimum_confidence)
        & (qwen_margin <= maximum_margin)
        & (calibrator_prediction != qwen_prediction)
    )

    result = qwen_prediction.copy()
    result[use_calibrator] = blended_prediction[use_calibrator]

    print(
        "Calibrator overrides:",
        int(use_calibrator.sum()),
    )
    return result


def main() -> None:
    started = time.perf_counter()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for Qwen inference."
        )

    required_paths = [
        QWEN_CONFIG_DIR,
        QWEN_INT8_DIR / "index.json",
        MODEL_DIR / "heads.pt",
        MODEL_DIR / "metadata.json",
        MODEL_DIR / "postprocess.json",
        TREE_ARTIFACTS_PATH,
        TREE_CONFIG_PATH,
        DATA_DIR / "test.jsonl",
        DATA_DIR / "sample_submission.csv",
    ]
    missing = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(missing)
        )

    metadata = json.loads(
        (MODEL_DIR / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    postprocess = json.loads(
        (MODEL_DIR / "postprocess.json").read_text(
            encoding="utf-8"
        )
    )
    tree_bundle = joblib.load(TREE_ARTIFACTS_PATH)
    tree_config = json.loads(TREE_CONFIG_PATH.read_text(encoding="utf-8"))
    tree_blend_weight = float(tree_config.get("blend_weight", tree_bundle.get("config", {}).get("blend_weight", 0.25)))
    print(f"Tree blend weight: {tree_blend_weight:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(QWEN_CONFIG_DIR),
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = load_int8_packed_qwen(
        QWEN_CONFIG_DIR,
        QWEN_INT8_DIR,
        tokenizer.pad_token_id,
    )

    model = QwenSegmentClassifier(
        backbone=backbone,
        hidden_size=backbone.config.hidden_size,
        structured_dim=int(
            metadata.get("structured_dim", STRUCTURED_DIM)
        ),
        num_labels=len(ALL_CLASSES),
        num_families=len(FAMILY_NAMES),
    )
    model.load_state_dict(
        safe_torch_load(MODEL_DIR / "heads.pt"),
        strict=False,
    )
    model.eval().to("cuda")

    samples = load_jsonl(DATA_DIR / "test.jsonl")
    ids = [
        str(sample.get("id", ""))
        for sample in samples
    ]

    dataset = EncodedDataset(
        samples,
        tokenizer,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=Collator(tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=True,
    )

    action_logits_parts: List[np.ndarray] = []
    family_logits_parts: List[np.ndarray] = []

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc="Qwen inference",
        ):
            batch = {
                key: value.to(
                    "cuda",
                    non_blocking=True,
                )
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                action_logits, family_logits = model(**batch)

            action_logits_parts.append(
                action_logits.float().cpu().numpy()
            )
            family_logits_parts.append(
                family_logits.float().cpu().numpy()
            )

    action_logits = np.concatenate(
        action_logits_parts,
        axis=0,
    ).astype(np.float64)
    family_logits = np.concatenate(
        family_logits_parts,
        axis=0,
    ).astype(np.float64)

    family_index = np.asarray(
        ACTION_TO_FAMILY,
        dtype=np.int64,
    )
    training_class_weights = np.asarray(
        postprocess["training_class_weights"],
        dtype=np.float64,
    )

    final_logits = (
        action_logits
        / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"])
        * family_logits[:, family_index]
        - float(postprocess["prior_beta"])
        * np.log(
            np.maximum(training_class_weights, 1e-12)
        )[None, :]
    )

    qwen_probabilities = stable_softmax(final_logits)
    tree_probabilities = predict_tree_probabilities(
        tree_bundle,
        samples,
    )

    blended_log_probability = (
        (1.0 - tree_blend_weight)
        * np.log(np.maximum(qwen_probabilities, 1e-12))
        + tree_blend_weight
        * np.log(np.maximum(tree_probabilities, 1e-12))
    )
    predictions = blended_log_probability.argmax(axis=1)

    print(
        "Tree blend predictions:",
        f"rows={len(predictions)}",
        f"weight={tree_blend_weight:.3f}",
    )

    prediction_map = {
        sample_id: ALL_CLASSES[int(prediction)]
        for sample_id, prediction in zip(ids, predictions)
    }

    with (
        DATA_DIR / "sample_submission.csv"
    ).open(
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or "id" not in fieldnames or "action" not in fieldnames:
        raise RuntimeError(
            "sample_submission.csv must contain id and action columns."
        )

    for row in rows:
        sample_id = str(row["id"])
        if sample_id not in prediction_map:
            raise KeyError(
                f"Missing prediction for id={sample_id}"
            )
        row["action"] = prediction_map[sample_id]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    print(
        f"Saved: {OUTPUT_PATH} rows={len(rows)}"
    )
    print(f"Elapsed seconds: {elapsed:.3f}")


if __name__ == "__main__":
    main()
