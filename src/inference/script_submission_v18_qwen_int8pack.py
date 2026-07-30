import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
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
        MODEL_DIR / "calibrator.joblib",
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
    calibrator_bundle = joblib.load(
        MODEL_DIR / "calibrator.joblib"
    )

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
    calibrator_features = build_calibrator_features(
        samples,
        action_logits,
        family_logits,
        final_logits,
    )

    expected_feature_count = len(
        calibrator_bundle.get("feature_names", [])
    )
    if (
        expected_feature_count
        and calibrator_features.shape[1]
        != expected_feature_count
    ):
        raise RuntimeError(
            "Calibrator feature dimension mismatch: "
            f"{calibrator_features.shape[1]} != "
            f"{expected_feature_count}"
        )

    calibrator_model = calibrator_bundle["model"]
    calibrator_probabilities = aligned_predict_proba(
        calibrator_model,
        calibrator_features,
    )
    config = calibrator_bundle["pooled_config"]

    predictions = apply_calibrator(
        qwen_probabilities,
        calibrator_probabilities,
        config,
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
