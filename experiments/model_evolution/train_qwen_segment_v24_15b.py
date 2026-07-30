from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from feature_utils_qwen_v4 import (
    ACTION_TO_FAMILY,
    ALL_CLASSES,
    FAMILY_NAMES,
    STRUCTURED_DIM,
    build_segments,
    build_structured_features,
)


LABEL2ID = {label: index for index, label in enumerate(ALL_CLASSES)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
NUM_CLASSES = len(ALL_CLASSES)
SEED = 42

BASE_SEGMENT_BUDGETS = {
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
FAMILY_LOGIT_WEIGHTS = [round(x * 0.05, 2) for x in range(0, 21)]
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-Coder 1.5B segment classifier fold/full trainer."
    )
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("model/qwen_segment_v24_15b_eval"))
    parser.add_argument("--mode", choices=["eval", "full"], default="eval")
    parser.add_argument("--eval-fold", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-per-class", type=int, default=80)
    parser.add_argument("--overfit-check", action="store_true")
    parser.add_argument("--overfit-per-class", type=int, default=10)

    # RTX 4060 8GB safe defaults. Increase only after one successful run.
    parser.add_argument("--max-length", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--max-updates", type=int, default=1600)
    parser.add_argument("--head-lr", type=float, default=4e-4)
    parser.add_argument("--lora-lr", type=float, default=1e-5)
    parser.add_argument("--family-loss-weight", type=float, default=0.25)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--tokenize-chunk-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(data_path: Path, labels_path: Path) -> Tuple[List[dict], List[str]]:
    samples: List[dict] = []
    with data_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    with labels_path.open(encoding="utf-8", newline="") as file:
        label_map = {str(row["id"]): str(row["action"]) for row in csv.DictReader(file)}

    missing = [str(sample.get("id", "")) for sample in samples if str(sample.get("id", "")) not in label_map]
    if missing:
        raise RuntimeError(f"Missing labels for {len(missing)} samples. First missing id={missing[0]}")

    targets = [label_map[str(sample["id"])] for sample in samples]
    unknown = sorted({target for target in targets if target not in LABEL2ID})
    if unknown:
        raise RuntimeError(f"Unknown target labels: {unknown}")
    return samples, targets


def balanced_subset(
    samples: Sequence[dict],
    targets: Sequence[str],
    per_class: int,
) -> Tuple[List[dict], List[str]]:
    rng = random.Random(SEED)
    buckets: Dict[str, List[int]] = {label: [] for label in ALL_CLASSES}
    for index, target in enumerate(targets):
        buckets[target].append(index)

    selected: List[int] = []
    for label in ALL_CLASSES:
        indices = buckets[label][:]
        rng.shuffle(indices)
        selected.extend(indices[: min(per_class, len(indices))])
    rng.shuffle(selected)
    return [samples[index] for index in selected], [targets[index] for index in selected]


def make_segment_budgets(max_length: int) -> Dict[str, int]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    total = sum(BASE_SEGMENT_BUDGETS.values())
    scaled = {
        name: max(8, int(round(value * max_length / total)))
        for name, value in BASE_SEGMENT_BUDGETS.items()
    }
    diff = max_length - sum(scaled.values())
    # Put rounding slack into the current prompt first because it is usually the
    # most predictive segment for the next action.
    order = ["current", "action", "history", "meta"]
    step = 1 if diff >= 0 else -1
    for i in range(abs(diff)):
        key = order[i % len(order)]
        if step < 0 and scaled[key] <= 8:
            continue
        scaled[key] += step
    return scaled


def smart_trim(token_ids: List[int], budget: int, prefix_tokens: int = 8) -> List[int]:
    if len(token_ids) <= budget:
        return token_ids
    prefix_tokens = min(prefix_tokens, max(0, budget // 3))
    if prefix_tokens == 0:
        return token_ids[-budget:]
    return token_ids[:prefix_tokens] + token_ids[-(budget - prefix_tokens):]


class EncodedSegmentDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[dict],
        labels: np.ndarray,
        tokenizer,
        max_length: int,
        segment_budgets: Dict[str, int],
        chunk_size: int,
        description: str,
    ):
        self.input_ids: List[np.ndarray] = []
        self.segment_ids: List[np.ndarray] = []
        self.structured_features: List[np.ndarray] = []
        self.labels = [int(label) for label in labels]
        self.family_labels = [int(ACTION_TO_FAMILY[int(label)]) for label in labels]

        segments = [build_segments(sample) for sample in samples]
        self.structured_features = [build_structured_features(sample) for sample in samples]

        for start in tqdm(range(0, len(samples), chunk_size), desc=description):
            batch_segments = segments[start:start + chunk_size]
            encoded_by_segment: Dict[str, List[List[int]]] = {}
            for segment_name in ("history", "action", "meta", "current"):
                encoded_by_segment[segment_name] = tokenizer(
                    [item[segment_name] for item in batch_segments],
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                )["input_ids"]

            for local_index in range(len(batch_segments)):
                combined_ids: List[int] = []
                combined_segment_ids: List[int] = []
                for segment_name in ("history", "action", "meta", "current"):
                    tokens = smart_trim(
                        encoded_by_segment[segment_name][local_index],
                        segment_budgets[segment_name],
                    )
                    combined_ids.extend(tokens)
                    combined_segment_ids.extend([SEGMENT_IDS[segment_name]] * len(tokens))

                if not combined_ids:
                    fallback = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
                    combined_ids = [fallback]
                    combined_segment_ids = [SEGMENT_IDS["current"]]

                self.input_ids.append(np.asarray(combined_ids[-max_length:], dtype=np.int32))
                self.segment_ids.append(np.asarray(combined_segment_ids[-max_length:], dtype=np.int8))

        if len(self.input_ids) != len(self.labels):
            raise RuntimeError("Tokenized sample count and label count differ.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {
            "input_ids": self.input_ids[index],
            "segment_ids": self.segment_ids[index],
            "structured_features": self.structured_features[index],
            "labels": self.labels[index],
            "family_labels": self.family_labels[index],
        }


class SegmentCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: List[dict]) -> dict:
        batch_size = len(features)
        max_length = max(len(feature["input_ids"]) for feature in features)
        padded_length = ((max_length + 7) // 8) * 8

        input_ids = torch.full((batch_size, padded_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, padded_length), dtype=torch.long)
        segment_ids = torch.full((batch_size, padded_length), -1, dtype=torch.long)
        structured_features = torch.tensor(
            np.stack([feature["structured_features"] for feature in features]),
            dtype=torch.float32,
        )
        labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
        family_labels = torch.tensor([feature["family_labels"] for feature in features], dtype=torch.long)

        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.from_numpy(feature["input_ids"].astype(np.int64, copy=False))
            attention_mask[row, :length] = 1
            segment_ids[row, :length] = torch.from_numpy(feature["segment_ids"].astype(np.int64, copy=False))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "segment_ids": segment_ids,
            "structured_features": structured_features,
            "labels": labels,
            "family_labels": family_labels,
        }


class SegmentProjector(nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(self.norm(values.float())))


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
        self.current_projector = SegmentProjector(hidden_size, projection_size)
        self.action_projector = SegmentProjector(hidden_size, projection_size)
        self.history_projector = SegmentProjector(hidden_size, projection_size)
        self.all_projector = SegmentProjector(hidden_size, projection_size)
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
    def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        current_pool = self.masked_mean(hidden, segment_ids.eq(SEGMENT_IDS["current"]))
        action_pool = self.masked_mean(hidden, segment_ids.eq(SEGMENT_IDS["action"]))
        history_pool = self.masked_mean(hidden, segment_ids.eq(SEGMENT_IDS["history"]))
        all_pool = self.masked_mean(hidden, attention_mask.bool())

        fused = torch.cat([
            self.current_projector(current_pool),
            self.action_projector(action_pool),
            self.history_projector(history_pool),
            self.all_projector(all_pool),
            self.structured_mlp(structured_features),
        ], dim=-1)
        representation = self.fusion(fused)
        return self.action_head(representation), self.family_head(representation)


def make_split(
    samples: Sequence[dict],
    targets: Sequence[str],
    eval_fold: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([LABEL2ID[target] for target in targets], dtype=np.int64)
    groups = np.asarray([str(sample["id"]).rsplit("-step_", 1)[0] for sample in samples])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (train_indices, validation_indices) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        if fold == eval_fold:
            overlap = set(groups[train_indices]) & set(groups[validation_indices])
            if overlap:
                raise RuntimeError(f"Session overlap detected: {len(overlap)}")
            return (
                train_indices.astype(np.int64),
                validation_indices.astype(np.int64),
                labels,
                groups,
            )
    raise ValueError(f"Invalid eval_fold={eval_fold}; expected 0..4")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    action_logits_all: List[np.ndarray] = []
    family_logits_all: List[np.ndarray] = []
    labels_all: List[int] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Validation", leave=False):
            labels = batch.pop("labels")
            batch.pop("family_labels")
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                action_logits, family_logits = model(**batch)
            action_logits_all.append(action_logits.float().cpu().numpy())
            family_logits_all.append(family_logits.float().cpu().numpy())
            labels_all.extend(labels.tolist())

    action_logits_np = np.concatenate(action_logits_all, axis=0).astype(np.float32)
    family_logits_np = np.concatenate(family_logits_all, axis=0).astype(np.float32)
    labels_np = np.asarray(labels_all, dtype=np.int64)
    family_index = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)

    best_score = -1.0
    best_weight = 0.0
    best_predictions = None
    for family_weight in FAMILY_LOGIT_WEIGHTS:
        final_logits = action_logits_np + family_weight * family_logits_np[:, family_index]
        predictions = final_logits.argmax(axis=1)
        score = f1_score(
            labels_np,
            predictions,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        )
        if score > best_score:
            best_score = float(score)
            best_weight = float(family_weight)
            best_predictions = predictions.astype(np.int64)

    return {
        "macro_f1": best_score,
        "family_logit_weight": best_weight,
        "labels": labels_np,
        "predictions": best_predictions,
        "action_logits": action_logits_np,
        "family_logits": family_logits_np,
    }


def configure_gradient_checkpointing(backbone, enabled: bool) -> None:
    if not enabled:
        return
    if hasattr(backbone, "gradient_checkpointing_enable"):
        try:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            backbone.gradient_checkpointing_enable()
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()


def build_model(args: argparse.Namespace, tokenizer, device: torch.device) -> QwenSegmentClassifier:
    print("Load Qwen backbone:", args.model_name)
    backbone = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    backbone.config.use_cache = False
    backbone.config.pad_token_id = tokenizer.pad_token_id
    configure_gradient_checkpointing(backbone, args.gradient_checkpointing)

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )
    backbone = get_peft_model(backbone, lora_config)

    # PEFT initializes LoRA weights in fp32-friendly form. Keeping trainable
    # adapter weights in fp32 is more stable than half-precision optimizer states.
    for parameter in backbone.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    model = QwenSegmentClassifier(
        backbone=backbone,
        hidden_size=backbone.config.hidden_size,
        structured_dim=STRUCTURED_DIM,
        num_labels=NUM_CLASSES,
        num_families=len(FAMILY_NAMES),
    ).to(device)
    return model


def make_loss_weights(train_labels: np.ndarray, device: torch.device, overfit_check: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if overfit_check:
        return (
            torch.ones(NUM_CLASSES, dtype=torch.float32, device=device),
            torch.ones(len(FAMILY_NAMES), dtype=torch.float32, device=device),
        )

    action_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float32)
    family_labels_np = np.asarray([ACTION_TO_FAMILY[int(label)] for label in train_labels], dtype=np.int64)
    family_counts = np.bincount(family_labels_np, minlength=len(FAMILY_NAMES)).astype(np.float32)

    action_weight_values = np.power(
        len(train_labels) / (NUM_CLASSES * np.maximum(action_counts, 1.0)),
        0.25,
    )
    family_weight_values = np.power(
        len(family_labels_np) / (len(FAMILY_NAMES) * np.maximum(family_counts, 1.0)),
        0.25,
    )
    return (
        torch.tensor(action_weight_values, dtype=torch.float32, device=device),
        torch.tensor(family_weight_values, dtype=torch.float32, device=device),
    )


def save_model(
    model: QwenSegmentClassifier,
    tokenizer,
    output_dir: Path,
    metadata: Dict[str, object],
    evaluation: Dict[str, object] | None,
    validation_indices: np.ndarray | None,
    class_counts: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.backbone.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(output_dir))

    heads = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("backbone.")
    }
    torch.save(heads, output_dir / "heads.pt")

    metadata_to_save = dict(metadata)
    if evaluation is not None:
        metadata_to_save["validation_macro_f1"] = float(evaluation["macro_f1"])
        metadata_to_save["family_logit_weight"] = float(evaluation["family_logit_weight"])
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata_to_save, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    training_class_weights = class_counts.astype(np.float64)
    training_class_weights = training_class_weights / max(float(training_class_weights.sum()), 1.0)
    postprocess = {
        "action_temperature": 1.0,
        "family_weight": None if evaluation is None else float(evaluation["family_logit_weight"]),
        "prior_beta": 0.0,
        "training_class_weights": training_class_weights.tolist(),
        "note": "V24 1.5B segment trainer uses direct action logits plus tuned family logits.",
    }
    (output_dir / "postprocess.json").write_text(
        json.dumps(postprocess, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if evaluation is not None:
        np.savez_compressed(
            output_dir / "validation_logits_v24_15b.npz",
            action_logits=evaluation["action_logits"],
            family_logits=evaluation["family_logits"],
            labels=evaluation["labels"],
            predictions=evaluation["predictions"],
            validation_indices=np.asarray([] if validation_indices is None else validation_indices, dtype=np.int64),
        )


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Load data...")
    samples, targets = load_data(args.data, args.labels)
    if args.overfit_check:
        samples, targets = balanced_subset(samples, targets, per_class=args.overfit_per_class)
        print("Overfit check samples:", len(samples))
    elif args.smoke:
        samples, targets = balanced_subset(samples, targets, per_class=args.smoke_per_class)
        print("Smoke samples:", len(samples))
    else:
        print("Full samples:", len(samples))

    print("Load tokenizer:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_labels = np.asarray([LABEL2ID[target] for target in targets], dtype=np.int64)
    if args.overfit_check:
        train_indices = np.arange(len(all_labels), dtype=np.int64)
        validation_indices = train_indices.copy()
        mode_for_metadata = "overfit_check"
    elif args.mode == "eval":
        train_indices, validation_indices, all_labels, _groups = make_split(samples, targets, args.eval_fold)
        mode_for_metadata = "eval"
    else:
        train_indices = np.arange(len(all_labels), dtype=np.int64)
        validation_indices = np.asarray([], dtype=np.int64)
        mode_for_metadata = "full"

    train_samples = [samples[int(index)] for index in train_indices]
    train_labels = all_labels[train_indices]
    validation_samples = [samples[int(index)] for index in validation_indices]
    validation_labels = all_labels[validation_indices]

    print(f"mode={mode_for_metadata} fold={args.eval_fold} train={len(train_samples)} val={len(validation_samples)}")
    print("Structured dim:", STRUCTURED_DIM)
    segment_budgets = make_segment_budgets(args.max_length)
    print("Max length:", args.max_length)
    print("Segment budgets:", segment_budgets)

    train_dataset = EncodedSegmentDataset(
        train_samples,
        train_labels,
        tokenizer,
        max_length=args.max_length,
        segment_budgets=segment_budgets,
        chunk_size=args.tokenize_chunk_size,
        description="Tokenize V24 1.5B train",
    )
    collator = SegmentCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    validation_loader = None
    if len(validation_samples) > 0:
        validation_dataset = EncodedSegmentDataset(
            validation_samples,
            validation_labels,
            tokenizer,
            max_length=args.max_length,
            segment_budgets=segment_budgets,
            chunk_size=args.tokenize_chunk_size,
            description="Tokenize V24 1.5B validation",
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.validation_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    model = build_model(args, tokenizer, device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    action_weights, family_weights = make_loss_weights(train_labels, device, args.overfit_check)

    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    head_ids = {id(parameter) for parameter in head_parameters}
    lora_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    optimizer = torch.optim.AdamW([
        {"params": head_parameters, "lr": args.head_lr, "weight_decay": 0.01},
        {"params": lora_parameters, "lr": args.lora_lr, "weight_decay": 0.0},
    ])

    updates_per_epoch = max(1, (len(train_loader) + args.grad_accum - 1) // args.grad_accum)
    planned_updates = updates_per_epoch * args.epochs
    if args.max_updates and args.max_updates > 0:
        planned_updates = min(planned_updates, args.max_updates)
    warmup_steps = max(1, int(planned_updates * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_updates,
    )
    scaler = torch.amp.GradScaler("cuda")

    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float64)
    metadata: Dict[str, object] = {
        "architecture": "qwen_segment_v24_15b",
        "base_model": args.model_name,
        "classes": ALL_CLASSES,
        "families": FAMILY_NAMES,
        "action_to_family": ACTION_TO_FAMILY,
        "structured_dim": STRUCTURED_DIM,
        "max_length": int(args.max_length),
        "segment_budgets": segment_budgets,
        "mode": mode_for_metadata,
        "eval_fold": int(args.eval_fold),
        "train_samples": int(len(train_samples)),
        "validation_samples": int(len(validation_samples)),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "lora": {
            "r": int(args.lora_r),
            "alpha": int(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "target_modules": LORA_TARGET_MODULES,
        },
        "learning_rates": {
            "head_lr": float(args.head_lr),
            "lora_lr": float(args.lora_lr),
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "validation_batch_size": int(args.validation_batch_size),
            "grad_accum": int(args.grad_accum),
            "max_updates": int(args.max_updates),
            "planned_updates": int(planned_updates),
            "family_loss_weight": float(args.family_loss_weight),
        },
        "epoch_records": [],
    }

    best_eval: Dict[str, object] | None = None
    best_epoch = 0
    best_true = None
    best_pred = None
    global_update = 0
    stop_training = False
    optimizer.zero_grad(set_to_none=True)

    print(f"Start training: epochs={args.epochs}, planned_updates={planned_updates}, grad_accum={args.grad_accum}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_action = 0.0
        running_family = 0.0
        batch_count = 0
        progress = tqdm(train_loader, desc=f"V24 1.5B epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(progress, start=1):
            labels_batch = batch.pop("labels").to(device, non_blocking=True)
            family_labels_batch = batch.pop("family_labels").to(device, non_blocking=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                action_logits, family_logits = model(**batch)
                action_loss = F.cross_entropy(action_logits.float(), labels_batch, weight=action_weights)
                family_loss = F.cross_entropy(family_logits.float(), family_labels_batch, weight=family_weights)
                raw_loss = action_loss + args.family_loss_weight * family_loss
                loss = raw_loss / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += float(raw_loss.detach().item())
            running_action += float(action_loss.detach().item())
            running_family += float(family_loss.detach().item())
            batch_count += 1

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=args.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                if args.max_updates and global_update >= args.max_updates:
                    stop_training = True

            if step % 100 == 0:
                progress.set_postfix(
                    loss=f"{raw_loss.detach().item():.4f}",
                    updates=global_update,
                )
            if stop_training:
                break

        record: Dict[str, object] = {
            "epoch": int(epoch),
            "global_update": int(global_update),
            "loss": float(running_loss / max(1, batch_count)),
            "action_loss": float(running_action / max(1, batch_count)),
            "family_loss": float(running_family / max(1, batch_count)),
        }

        if validation_loader is not None:
            evaluation = evaluate(model, validation_loader, device)
            record["macro_f1"] = float(evaluation["macro_f1"])
            record["family_logit_weight"] = float(evaluation["family_logit_weight"])
            print(
                f"V24 1.5B epoch {epoch} "
                f"macro_f1={evaluation['macro_f1']:.6f} "
                f"family_logit_weight={evaluation['family_logit_weight']:.2f} "
                f"loss={record['loss']:.6f}"
            )
            if best_eval is None or float(evaluation["macro_f1"]) > float(best_eval["macro_f1"]):
                best_eval = evaluation
                best_epoch = epoch
                best_true = evaluation["labels"]
                best_pred = evaluation["predictions"]
                metadata["best_epoch"] = int(best_epoch)
                metadata["epoch_records"].append(record)
                save_model(model, tokenizer, args.output_dir, metadata, best_eval, validation_indices, class_counts)
            else:
                metadata["epoch_records"].append(record)
        else:
            print(f"V24 1.5B epoch {epoch} updates={global_update} loss={record['loss']:.6f}")
            metadata["best_epoch"] = int(epoch)
            metadata["epoch_records"].append(record)
            save_model(model, tokenizer, args.output_dir, metadata, None, None, class_counts)

        if stop_training:
            print(f"Reached max_updates={args.max_updates}.")
            break

    if best_eval is not None:
        print()
        print(f"Best V24 1.5B Macro-F1: {best_eval['macro_f1']:.6f}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best family logit weight: {best_eval['family_logit_weight']:.2f}")
        report = classification_report(
            best_true,
            best_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=ALL_CLASSES,
            digits=6,
            zero_division=0,
        )
        print(report)
        (args.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

        final_metadata = json.loads((args.output_dir / "metadata.json").read_text(encoding="utf-8"))
        final_metadata["final_summary"] = {
            "best_macro_f1": float(best_eval["macro_f1"]),
            "best_epoch": int(best_epoch),
            "best_family_logit_weight": float(best_eval["family_logit_weight"]),
            "global_updates": int(global_update),
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(final_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        final_metadata = json.loads((args.output_dir / "metadata.json").read_text(encoding="utf-8"))
        final_metadata["final_summary"] = {
            "mode": "full",
            "global_updates": int(global_update),
            "note": "Full training has no local validation score. Use only after eval-mode validation is promising.",
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(final_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("Saved:", args.output_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
