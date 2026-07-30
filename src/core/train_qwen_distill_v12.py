from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

import train_qwen_segment_v4 as v4
from feature_utils_qwen_v4 import (
    ACTION_TO_FAMILY,
    ALL_CLASSES,
    FAMILY_NAMES,
    STRUCTURED_DIM,
)


SEED = 42
NUM_CLASSES = len(ALL_CLASSES)
LABEL2ID = {label: i for i, label in enumerate(ALL_CLASSES)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
FAMILY_INDEX = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)


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
        self.current_projector = v4.SegmentProjector(hidden_size, projection_size)
        self.action_projector = v4.SegmentProjector(hidden_size, projection_size)
        self.history_projector = v4.SegmentProjector(hidden_size, projection_size)
        self.all_projector = v4.SegmentProjector(hidden_size, projection_size)

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

        current_pool = self.masked_mean(
            hidden,
            segment_ids.eq(v4.SEGMENT_IDS["current"]),
        )
        action_pool = self.masked_mean(
            hidden,
            segment_ids.eq(v4.SEGMENT_IDS["action"]),
        )
        history_pool = self.masked_mean(
            hidden,
            segment_ids.eq(v4.SEGMENT_IDS["history"]),
        )
        all_pool = self.masked_mean(hidden, attention_mask.bool())

        fused = torch.cat([
            self.current_projector(current_pool),
            self.action_projector(action_pool),
            self.history_projector(history_pool),
            self.all_projector(all_pool),
            self.structured_mlp(structured_features),
        ], dim=-1)

        fused = self.fusion(fused)
        return self.action_head(fused), self.family_head(fused)


class DistillDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        teacher_action_logits: np.ndarray,
        teacher_family_logits: np.ndarray,
        teacher_final_logits: np.ndarray,
    ):
        self.base_dataset = base_dataset
        self.teacher_action_logits = teacher_action_logits.astype(np.float32)
        self.teacher_family_logits = teacher_family_logits.astype(np.float32)
        self.teacher_final_logits = teacher_final_logits.astype(np.float32)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.base_dataset[index])
        item["teacher_action_logits"] = torch.tensor(
            self.teacher_action_logits[index],
            dtype=torch.float32,
        )
        item["teacher_family_logits"] = torch.tensor(
            self.teacher_family_logits[index],
            dtype=torch.float32,
        )
        item["teacher_final_logits"] = torch.tensor(
            self.teacher_final_logits[index],
            dtype=torch.float32,
        )
        return item


class DistillCollator:
    def __init__(self, pad_token_id: int):
        self.base_collator = v4.SegmentCollator(pad_token_id)

    def __call__(self, features: List[dict]) -> dict:
        teacher_action = torch.stack([f.pop("teacher_action_logits") for f in features])
        teacher_family = torch.stack([f.pop("teacher_family_logits") for f in features])
        teacher_final = torch.stack([f.pop("teacher_final_logits") for f in features])
        batch = self.base_collator(features)
        batch["teacher_action_logits"] = teacher_action
        batch["teacher_family_logits"] = teacher_family
        batch["teacher_final_logits"] = teacher_final
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V12 OOF-Teacher Distilled Qwen. This keeps the V4 submission "
            "architecture but changes the training objective: hard labels plus "
            "OOF teacher soft targets from the 5-fold Qwen ensemble."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--v4-dir", type=Path, default=Path("model/qwen_segment_v4"))
    parser.add_argument("--oof-logits", type=Path, default=Path("model/qwen_v4_oof/oof_logits_all_70000.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/qwen_distill_v12"))
    parser.add_argument("--mode", choices=["eval", "full"], default="eval")
    parser.add_argument("--eval-fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--head-lr", type=float, default=4e-4)
    parser.add_argument("--lora-lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--tokenize-chunk-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)

    # Loss weights. Conservative by default: do not let teacher overwrite labels.
    parser.add_argument("--hard-final-weight", type=float, default=0.70)
    parser.add_argument("--hard-action-weight", type=float, default=0.35)
    parser.add_argument("--family-weight", type=float, default=0.25)
    parser.add_argument("--distill-final-weight", type=float, default=0.55)
    parser.add_argument("--distill-action-weight", type=float, default=0.15)
    parser.add_argument("--distill-family-weight", type=float, default=0.10)
    parser.add_argument("--teacher-ce-weight", type=float, default=0.10)
    parser.add_argument("--distill-temperature", type=float, default=2.0)

    parser.add_argument("--freeze-lora", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(data_path: Path, labels_path: Path) -> Tuple[List[dict], np.ndarray]:
    samples: List[dict] = []
    with data_path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                samples.append(json.loads(line))

    with labels_path.open(encoding="utf-8", newline="") as file:
        label_map = {str(row["id"]): LABEL2ID[str(row["action"])] for row in csv.DictReader(file)}

    labels = np.asarray([label_map[str(sample["id"])] for sample in samples], dtype=np.int64)
    return samples, labels


def make_split(samples: Sequence[dict], labels: np.ndarray, eval_fold: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray([
        str(sample["id"]).rsplit("-step_", 1)[0]
        for sample in samples
    ])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups)
    ):
        if fold == eval_fold:
            overlap = set(groups[train_idx]) & set(groups[val_idx])
            if overlap:
                raise RuntimeError(f"Session overlap detected: {len(overlap)}")
            return train_idx.astype(np.int64), val_idx.astype(np.int64), groups

    raise ValueError(f"Invalid eval fold: {eval_fold}")


def balanced_smoke_indices(labels: np.ndarray, per_class: int = 100) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    selected: List[int] = []
    for label in range(NUM_CLASSES):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        selected.extend(indices[: min(per_class, len(indices))].tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def final_logits_numpy(
    action_logits: np.ndarray,
    family_logits: np.ndarray,
    class_weights: np.ndarray,
    postprocess: Dict[str, Any],
) -> np.ndarray:
    return (
        action_logits.astype(np.float64) / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"]) * family_logits.astype(np.float64)[:, FAMILY_INDEX]
        - float(postprocess["prior_beta"]) * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )


def final_logits_torch(
    action_logits: torch.Tensor,
    family_logits: torch.Tensor,
    class_weights: torch.Tensor,
    family_index: torch.Tensor,
    postprocess: Dict[str, Any],
) -> torch.Tensor:
    return (
        action_logits.float() / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"]) * family_logits.float()[:, family_index]
        - float(postprocess["prior_beta"]) * torch.log(torch.clamp(class_weights, min=1e-12))[None, :]
    )


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        f1_score(
            labels,
            predictions,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0,
        )
    )


def class_f1_values(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return f1_score(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
        average=None,
        zero_division=0,
    )


def load_model_from_v4(
    v4_dir: Path,
    model_name: str,
    tokenizer,
    device: torch.device,
    train_lora: bool,
) -> QwenSegmentClassifier:
    base_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id

    backbone = PeftModel.from_pretrained(
        base_model,
        str(v4_dir / "adapter"),
        is_trainable=train_lora,
    )

    model = QwenSegmentClassifier(
        backbone=backbone,
        hidden_size=backbone.config.hidden_size,
        structured_dim=STRUCTURED_DIM,
        num_labels=NUM_CLASSES,
        num_families=len(FAMILY_NAMES),
    )

    result = model.load_state_dict(
        safe_torch_load(v4_dir / "heads.pt"),
        strict=False,
    )
    unexpected = list(result.unexpected_keys)
    missing_non_backbone = [
        key for key in result.missing_keys
        if not key.startswith("backbone.")
    ]
    if unexpected or missing_non_backbone:
        raise RuntimeError(
            f"Bad V4 head load. missing_non_backbone={missing_non_backbone}, "
            f"unexpected={unexpected}"
        )
    print(
        f"Loaded V4 heads. Ignored backbone missing keys: "
        f"{len(result.missing_keys) - len(missing_non_backbone)}"
    )

    model.to(device)
    return model


def build_optimizer(model: QwenSegmentClassifier, args: argparse.Namespace):
    head_params = []
    lora_params = []

    for name, parameter in model.named_parameters():
        parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        if name.startswith("backbone."):
            if "lora_" in name and not args.freeze_lora:
                parameter.requires_grad = True
                lora_params.append(parameter)
        else:
            parameter.requires_grad = True
            head_params.append(parameter)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": args.head_lr, "weight_decay": 0.01})
    if lora_params:
        groups.append({"params": lora_params, "lr": args.lora_lr, "weight_decay": 0.0})

    if not groups:
        raise RuntimeError("No trainable parameters.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    print(f"Optimizer groups: heads={len(head_params)} tensors, lora={len(lora_params)} tensors")
    return torch.optim.AdamW(groups)


def kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student = student_logits.float() / temperature
    teacher = teacher_logits.float() / temperature
    return (
        F.kl_div(
            F.log_softmax(student, dim=-1),
            F.softmax(teacher, dim=-1),
            reduction="batchmean",
        )
        * (temperature ** 2)
    )


def evaluate(
    model: QwenSegmentClassifier,
    loader: DataLoader,
    device: torch.device,
    postprocess: Dict[str, Any],
    class_weights_tensor: torch.Tensor,
    family_index_tensor: torch.Tensor,
) -> Dict[str, Any]:
    model.eval()
    action_parts: List[np.ndarray] = []
    family_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Validation", leave=False):
            labels = batch.pop("labels")
            batch.pop("family_labels")
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

            # Remove teacher keys if this is a DistillDataset batch.
            batch.pop("teacher_action_logits", None)
            batch.pop("teacher_family_logits", None)
            batch.pop("teacher_final_logits", None)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                action_logits, family_logits = model(**batch)

            action_parts.append(action_logits.float().cpu().numpy())
            family_parts.append(family_logits.float().cpu().numpy())
            label_parts.append(labels.numpy())

    action_logits_np = np.concatenate(action_parts, axis=0).astype(np.float32)
    family_logits_np = np.concatenate(family_parts, axis=0).astype(np.float32)
    labels_np = np.concatenate(label_parts, axis=0).astype(np.int64)
    class_weights = class_weights_tensor.detach().cpu().numpy().astype(np.float64)

    final_np = final_logits_numpy(action_logits_np, family_logits_np, class_weights, postprocess)
    predictions = final_np.argmax(axis=1).astype(np.int64)
    score = macro_f1(labels_np, predictions)

    return {
        "macro_f1": score,
        "labels": labels_np,
        "predictions": predictions,
        "action_logits": action_logits_np,
        "family_logits": family_logits_np,
        "class_f1": class_f1_values(labels_np, predictions),
    }


def save_model(
    model: QwenSegmentClassifier,
    tokenizer,
    output_dir: Path,
    metadata: Dict[str, Any],
    postprocess: Dict[str, Any],
    evaluation: Dict[str, Any] | None = None,
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

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata_to_save, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "postprocess.json").write_text(
        json.dumps(postprocess, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if evaluation is not None:
        np.savez_compressed(
            output_dir / "validation_logits_v12.npz",
            action_logits=evaluation["action_logits"],
            family_logits=evaluation["family_logits"],
            labels=evaluation["labels"],
            predictions=evaluation["predictions"],
        )


def make_datasets(
    samples: Sequence[dict],
    labels: np.ndarray,
    indices: np.ndarray,
    tokenizer,
    teacher_action_logits: np.ndarray,
    teacher_family_logits: np.ndarray,
    teacher_final_logits: np.ndarray,
    chunk_size: int,
    description: str,
) -> DistillDataset:
    subset_samples = [samples[int(i)] for i in indices]
    subset_labels = labels[indices]
    base_dataset = v4.EncodedSegmentDataset(
        subset_samples,
        subset_labels,
        tokenizer,
        chunk_size=chunk_size,
        description=description,
    )
    return DistillDataset(
        base_dataset,
        teacher_action_logits[indices],
        teacher_family_logits[indices],
        teacher_final_logits[indices],
    )


def make_plain_dataset(
    samples: Sequence[dict],
    labels: np.ndarray,
    indices: np.ndarray,
    tokenizer,
    chunk_size: int,
    description: str,
):
    subset_samples = [samples[int(i)] for i in indices]
    subset_labels = labels[indices]
    return v4.EncodedSegmentDataset(
        subset_samples,
        subset_labels,
        tokenizer,
        chunk_size=chunk_size,
        description=description,
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
    print("Load data and OOF teacher logits...")
    samples, labels = load_data(args.data, args.labels)
    oof = np.load(args.oof_logits)

    teacher_action_logits = oof["action_logits"].astype(np.float32)
    teacher_family_logits = oof["family_logits"].astype(np.float32)
    oof_labels = oof["labels"].astype(np.int64)

    if not np.array_equal(labels, oof_labels):
        raise RuntimeError("OOF labels do not align with labels file.")

    metadata_v4 = json.loads((args.v4_dir / "metadata.json").read_text(encoding="utf-8"))
    postprocess = json.loads((args.v4_dir / "postprocess.json").read_text(encoding="utf-8"))
    class_weights_np = np.asarray(postprocess["training_class_weights"], dtype=np.float64)
    teacher_final_logits = final_logits_numpy(
        teacher_action_logits,
        teacher_family_logits,
        class_weights_np,
        postprocess,
    ).astype(np.float32)

    if args.smoke:
        selected = balanced_smoke_indices(labels, per_class=100)
        samples = [samples[int(i)] for i in selected]
        labels = labels[selected]
        teacher_action_logits = teacher_action_logits[selected]
        teacher_family_logits = teacher_family_logits[selected]
        teacher_final_logits = teacher_final_logits[selected]
        args.epochs = 1
        args.max_updates = min(args.max_updates or 80, 80)
        print("Smoke samples:", len(samples))

    if args.mode == "eval":
        train_idx, val_idx, groups = make_split(samples, labels, args.eval_fold)
        print(
            f"Mode=eval fold={args.eval_fold} "
            f"train={len(train_idx)} validation={len(val_idx)}"
        )
    else:
        train_idx = np.arange(len(labels), dtype=np.int64)
        val_idx = np.asarray([], dtype=np.int64)
        print(f"Mode=full train={len(train_idx)} validation=none")

    model_name = str(metadata_v4["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.v4_dir),
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenize datasets...")
    train_dataset = make_datasets(
        samples,
        labels,
        train_idx,
        tokenizer,
        teacher_action_logits,
        teacher_family_logits,
        teacher_final_logits,
        chunk_size=args.tokenize_chunk_size,
        description="Tokenize V12 train",
    )
    distill_collator = DistillCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=distill_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    validation_loader = None
    if args.mode == "eval":
        validation_dataset = make_plain_dataset(
            samples,
            labels,
            val_idx,
            tokenizer,
            chunk_size=args.tokenize_chunk_size,
            description="Tokenize V12 validation",
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=max(args.validation_batch_size, args.batch_size * 2),
            shuffle=False,
            collate_fn=v4.SegmentCollator(tokenizer.pad_token_id),
            num_workers=args.num_workers,
            pin_memory=True,
        )

    print("Load V12 student initialized from V4...")
    model = load_model_from_v4(
        args.v4_dir,
        model_name,
        tokenizer,
        device,
        train_lora=not args.freeze_lora,
    )

    optimizer = build_optimizer(model, args)

    class_counts = np.bincount(labels[train_idx], minlength=NUM_CLASSES).astype(np.float64)
    ce_weights = np.power(
        len(train_idx) / (NUM_CLASSES * np.maximum(class_counts, 1.0)),
        0.25,
    ).astype(np.float32)

    class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    family_index_tensor = torch.tensor(FAMILY_INDEX, dtype=torch.long, device=device)
    ce_weights_tensor = torch.tensor(ce_weights, dtype=torch.float32, device=device)

    total_batches = len(train_loader)
    updates_per_epoch = max(1, (total_batches + args.grad_accum - 1) // args.grad_accum)
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

    initial_eval = None
    best_eval = None
    best_epoch = 0
    if validation_loader is not None:
        initial_eval = evaluate(
            model,
            validation_loader,
            device,
            postprocess,
            class_weights_tensor,
            family_index_tensor,
        )
        best_eval = initial_eval
        print(f"Initial V4 Macro-F1: {initial_eval['macro_f1']:.6f}")

    metadata = {
        "architecture": "qwen_distill_v12",
        "base_model": model_name,
        "initialized_from": str(args.v4_dir),
        "classes": ALL_CLASSES,
        "families": FAMILY_NAMES,
        "structured_dim": STRUCTURED_DIM,
        "mode": args.mode,
        "eval_fold": int(args.eval_fold),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "loss_weights": {
            "hard_final": float(args.hard_final_weight),
            "hard_action": float(args.hard_action_weight),
            "family": float(args.family_weight),
            "distill_final": float(args.distill_final_weight),
            "distill_action": float(args.distill_action_weight),
            "distill_family": float(args.distill_family_weight),
            "teacher_ce": float(args.teacher_ce_weight),
            "temperature": float(args.distill_temperature),
        },
        "learning_rates": {
            "head_lr": float(args.head_lr),
            "lora_lr": float(args.lora_lr),
            "freeze_lora": bool(args.freeze_lora),
        },
        "initial_validation_macro_f1": None if initial_eval is None else float(initial_eval["macro_f1"]),
        "epoch_records": [],
        "best_epoch": 0,
    }

    if validation_loader is not None:
        save_model(model, tokenizer, args.output_dir, metadata, postprocess, initial_eval)

    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    stop_training = False

    print(
        f"Start V12 distillation: epochs={args.epochs}, "
        f"planned_updates={planned_updates}, grad_accum={args.grad_accum}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {
            "total": 0.0,
            "hard_final": 0.0,
            "hard_action": 0.0,
            "family": 0.0,
            "distill_final": 0.0,
            "distill_action": 0.0,
            "distill_family": 0.0,
            "teacher_ce": 0.0,
        }
        batch_count = 0
        progress = tqdm(train_loader, desc=f"V12 epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(progress, start=1):
            labels_tensor = batch.pop("labels").to(device, non_blocking=True)
            family_labels = batch.pop("family_labels").to(device, non_blocking=True)
            teacher_action = batch.pop("teacher_action_logits").to(device, non_blocking=True)
            teacher_family = batch.pop("teacher_family_logits").to(device, non_blocking=True)
            teacher_final = batch.pop("teacher_final_logits").to(device, non_blocking=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                action_logits, family_logits = model(**batch)

            student_final = final_logits_torch(
                action_logits,
                family_logits,
                class_weights_tensor,
                family_index_tensor,
                postprocess,
            )

            hard_final_loss = F.cross_entropy(
                student_final,
                labels_tensor,
                weight=ce_weights_tensor,
            )
            hard_action_loss = F.cross_entropy(
                action_logits.float(),
                labels_tensor,
                weight=ce_weights_tensor,
            )
            family_loss = F.cross_entropy(
                family_logits.float(),
                family_labels,
            )
            distill_final_loss = kl_divergence(
                student_final,
                teacher_final,
                args.distill_temperature,
            )
            distill_action_loss = kl_divergence(
                action_logits,
                teacher_action,
                args.distill_temperature,
            )
            distill_family_loss = kl_divergence(
                family_logits,
                teacher_family,
                args.distill_temperature,
            )

            teacher_targets = teacher_final.argmax(dim=1)
            teacher_confidence = F.softmax(teacher_final.float(), dim=1).max(dim=1).values.detach()
            teacher_ce_unreduced = F.cross_entropy(
                student_final,
                teacher_targets,
                reduction="none",
            )
            teacher_ce_loss = (teacher_ce_unreduced * teacher_confidence).mean()

            raw_loss = (
                args.hard_final_weight * hard_final_loss
                + args.hard_action_weight * hard_action_loss
                + args.family_weight * family_loss
                + args.distill_final_weight * distill_final_loss
                + args.distill_action_weight * distill_action_loss
                + args.distill_family_weight * distill_family_loss
                + args.teacher_ce_weight * teacher_ce_loss
            )
            loss = raw_loss / args.grad_accum

            scaler.scale(loss).backward()

            running["total"] += float(raw_loss.detach().item())
            running["hard_final"] += float(hard_final_loss.detach().item())
            running["hard_action"] += float(hard_action_loss.detach().item())
            running["family"] += float(family_loss.detach().item())
            running["distill_final"] += float(distill_final_loss.detach().item())
            running["distill_action"] += float(distill_action_loss.detach().item())
            running["distill_family"] += float(distill_family_loss.detach().item())
            running["teacher_ce"] += float(teacher_ce_loss.detach().item())
            batch_count += 1

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
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
                    hard=f"{hard_final_loss.detach().item():.3f}",
                    kd=f"{distill_final_loss.detach().item():.3f}",
                    updates=global_update,
                )

            if stop_training:
                break

        record = {
            "epoch": int(epoch),
            "global_update": int(global_update),
            "loss": {key: value / max(1, batch_count) for key, value in running.items()},
        }

        if validation_loader is not None:
            evaluation = evaluate(
                model,
                validation_loader,
                device,
                postprocess,
                class_weights_tensor,
                family_index_tensor,
            )
            record["macro_f1"] = float(evaluation["macro_f1"])
            record["improvement"] = float(evaluation["macro_f1"] - initial_eval["macro_f1"])
            print(
                f"V12 epoch {epoch} "
                f"macro_f1={evaluation['macro_f1']:.6f} "
                f"improvement={evaluation['macro_f1'] - initial_eval['macro_f1']:+.6f} "
                f"loss={record['loss']['total']:.6f}"
            )

            if evaluation["macro_f1"] > best_eval["macro_f1"]:
                best_eval = evaluation
                best_epoch = epoch
                metadata["best_epoch"] = int(epoch)
                metadata["epoch_records"].append(record)
                save_model(model, tokenizer, args.output_dir, metadata, postprocess, evaluation)
            else:
                metadata["epoch_records"].append(record)
        else:
            print(
                f"V12 epoch {epoch} "
                f"updates={global_update} "
                f"loss={record['loss']['total']:.6f}"
            )
            metadata["epoch_records"].append(record)
            metadata["best_epoch"] = int(epoch)
            save_model(model, tokenizer, args.output_dir, metadata, postprocess, None)

        if stop_training:
            print(f"Reached max_updates={args.max_updates}.")
            break

    if validation_loader is not None:
        print()
        print(f"Initial Macro-F1: {initial_eval['macro_f1']:.6f}")
        print(f"Best V12 Macro-F1: {best_eval['macro_f1']:.6f}")
        print(f"Improvement: {best_eval['macro_f1'] - initial_eval['macro_f1']:+.6f}")
        print(f"Best epoch: {best_epoch}")

        print()
        print("Class F1 changes:")
        initial_f1 = initial_eval["class_f1"]
        best_f1 = best_eval["class_f1"]
        for index, label in enumerate(ALL_CLASSES):
            print(
                f"{label:18s} "
                f"{initial_f1[index]:.6f} -> {best_f1[index]:.6f} "
                f"({best_f1[index] - initial_f1[index]:+.6f})"
            )

        report = classification_report(
            best_eval["labels"],
            best_eval["predictions"],
            labels=np.arange(NUM_CLASSES),
            target_names=ALL_CLASSES,
            digits=6,
            zero_division=0,
        )
        print()
        print(report)
        (args.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

        final_metadata = json.loads((args.output_dir / "metadata.json").read_text(encoding="utf-8"))
        final_metadata["final_summary"] = {
            "initial_macro_f1": float(initial_eval["macro_f1"]),
            "best_macro_f1": float(best_eval["macro_f1"]),
            "improvement": float(best_eval["macro_f1"] - initial_eval["macro_f1"]),
            "best_epoch": int(best_epoch),
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
            "note": "Full training has no local validation score. Submit only after eval-mode validation is promising.",
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
