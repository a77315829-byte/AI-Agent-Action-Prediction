import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from feature_utils_qwen_v4 import (
    ACTION_TO_FAMILY,
    ALL_CLASSES,
    FAMILY_NAMES,
    STRUCTURED_DIM,
    build_segments,
    build_structured_features,
)


LABEL2ID = {
    label: index
    for index, label in enumerate(ALL_CLASSES)
}

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
SAVE_DIR = BASE_DIR / "model" / "qwen_segment_v4"

SEED = 42
MAX_LENGTH = 256

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

FAMILY_LOGIT_WEIGHTS = [
    0.0,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overfit-check", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=8e-4)
    parser.add_argument("--lora-lr", type=float, default=8e-5)
    parser.add_argument(
        "--family-loss-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--tokenize-chunk-size",
        type=int,
        default=2048,
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data() -> Tuple[List[dict], List[str]]:
    samples: List[dict] = []

    with (DATA_DIR / "train.jsonl").open(
        encoding="utf-8"
    ) as file:
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

    targets = [
        labels[sample["id"]]
        for sample in samples
    ]

    return samples, targets


def balanced_subset(
    samples: List[dict],
    targets: List[str],
    per_class: int,
) -> Tuple[List[dict], List[str]]:
    rng = random.Random(SEED)
    buckets: Dict[str, List[int]] = {
        label: [] for label in ALL_CLASSES
    }

    for index, target in enumerate(targets):
        buckets[target].append(index)

    selected: List[int] = []

    for label in ALL_CLASSES:
        indices = buckets[label][:]
        rng.shuffle(indices)
        selected.extend(indices[:per_class])

    rng.shuffle(selected)

    return (
        [samples[index] for index in selected],
        [targets[index] for index in selected],
    )


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


class EncodedSegmentDataset(Dataset):
    def __init__(
        self,
        samples: List[dict],
        labels: np.ndarray,
        tokenizer,
        chunk_size: int,
        description: str,
    ):
        self.input_ids: List[np.ndarray] = []
        self.segment_ids: List[np.ndarray] = []
        self.structured_features: List[np.ndarray] = []
        self.labels = [
            int(label) for label in labels
        ]
        self.family_labels = [
            int(ACTION_TO_FAMILY[int(label)])
            for label in labels
        ]

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
            desc=description,
        ):
            batch_segments = segments[
                start:start + chunk_size
            ]

            encoded_by_segment = {}

            for segment_name in (
                "history",
                "action",
                "meta",
                "current",
            ):
                encoded_by_segment[segment_name] = (
                    tokenizer(
                        [
                            item[segment_name]
                            for item in batch_segments
                        ],
                        add_special_tokens=False,
                        padding=False,
                        truncation=False,
                    )["input_ids"]
                )

            for local_index in range(
                len(batch_segments)
            ):
                combined_ids: List[int] = []
                combined_segment_ids: List[int] = []

                for segment_name in (
                    "history",
                    "action",
                    "meta",
                    "current",
                ):
                    tokens = encoded_by_segment[
                        segment_name
                    ][local_index]

                    tokens = smart_trim(
                        tokens,
                        SEGMENT_BUDGETS[
                            segment_name
                        ],
                    )

                    combined_ids.extend(tokens)
                    combined_segment_ids.extend([
                        SEGMENT_IDS[segment_name]
                    ] * len(tokens))

                if not combined_ids:
                    fallback = (
                        tokenizer.eos_token_id
                        if tokenizer.eos_token_id is not None
                        else tokenizer.pad_token_id
                    )
                    combined_ids = [fallback]
                    combined_segment_ids = [
                        SEGMENT_IDS["current"]
                    ]

                combined_ids = combined_ids[-MAX_LENGTH:]
                combined_segment_ids = (
                    combined_segment_ids[-MAX_LENGTH:]
                )

                self.input_ids.append(
                    np.asarray(
                        combined_ids,
                        dtype=np.int32,
                    )
                )
                self.segment_ids.append(
                    np.asarray(
                        combined_segment_ids,
                        dtype=np.int8,
                    )
                )

        if len(self.input_ids) != len(self.labels):
            raise RuntimeError(
                "토큰화 결과와 label 수가 다릅니다."
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {
            "input_ids": self.input_ids[index],
            "segment_ids": self.segment_ids[index],
            "structured_features": (
                self.structured_features[index]
            ),
            "labels": self.labels[index],
            "family_labels": (
                self.family_labels[index]
            ),
        }


class SegmentCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: List[dict]) -> dict:
        batch_size = len(features)
        max_length = max(
            len(feature["input_ids"])
            for feature in features
        )
        padded_length = (
            (max_length + 7) // 8
        ) * 8

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

        labels = torch.tensor(
            [
                feature["labels"]
                for feature in features
            ],
            dtype=torch.long,
        )
        family_labels = torch.tensor(
            [
                feature["family_labels"]
                for feature in features
            ],
            dtype=torch.long,
        )

        for row, feature in enumerate(features):
            length = len(feature["input_ids"])

            input_ids[row, :length] = torch.from_numpy(
                feature["input_ids"].astype(
                    np.int64,
                    copy=False,
                )
            )
            attention_mask[row, :length] = 1
            segment_ids[row, :length] = torch.from_numpy(
                feature["segment_ids"].astype(
                    np.int64,
                    copy=False,
                )
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "segment_ids": segment_ids,
            "structured_features": (
                structured_features
            ),
            "labels": labels,
            "family_labels": family_labels,
        }


class SegmentProjector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size,
            output_size,
        )
        self.activation = nn.GELU()

    def forward(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        values = self.norm(values.float())
        return self.activation(self.linear(values))


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

        self.action_head = nn.Linear(
            512,
            num_labels,
        )
        self.family_head = nn.Linear(
            512,
            num_families,
        )

    @staticmethod
    def masked_mean(
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_float = mask.unsqueeze(-1).to(
            hidden.dtype
        )
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
            segment_ids.eq(
                SEGMENT_IDS["current"]
            ),
        )
        action_pool = self.masked_mean(
            hidden,
            segment_ids.eq(
                SEGMENT_IDS["action"]
            ),
        )
        history_pool = self.masked_mean(
            hidden,
            segment_ids.eq(
                SEGMENT_IDS["history"]
            ),
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
            self.structured_mlp(
                structured_features
            ),
        ], dim=-1)

        representation = self.fusion(fused)

        return (
            self.action_head(representation),
            self.family_head(representation),
        )


def make_split(
    samples: List[dict],
    targets: List[str],
):
    labels = np.asarray(
        [
            LABEL2ID[target]
            for target in targets
        ],
        dtype=np.int64,
    )
    groups = np.asarray([
        str(sample["id"]).rsplit(
            "-step_",
            1,
        )[0]
        for sample in samples
    ])

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    train_indices, validation_indices = next(
        splitter.split(
            np.zeros(len(labels)),
            labels,
            groups,
        )
    )

    overlap = (
        set(groups[train_indices])
        & set(groups[validation_indices])
    )

    if overlap:
        raise RuntimeError(
            f"세션 중복: {len(overlap)}"
        )

    return train_indices, validation_indices, labels


def adjusted_action_logits(
    action_logits: torch.Tensor,
    family_logits: torch.Tensor,
    family_weight: float,
) -> torch.Tensor:
    family_index = torch.tensor(
        ACTION_TO_FAMILY,
        dtype=torch.long,
        device=action_logits.device,
    )

    return (
        action_logits
        + family_weight
        * family_logits[:, family_index]
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()

    action_logits_all: List[np.ndarray] = []
    family_logits_all: List[np.ndarray] = []
    labels_all: List[int] = []

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc="Validation",
            leave=False,
        ):
            labels = batch.pop("labels")
            batch.pop("family_labels")

            batch = {
                key: value.to(
                    device,
                    non_blocking=True,
                )
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                action_logits, family_logits = (
                    model(**batch)
                )

            action_logits_all.append(
                action_logits.float().cpu().numpy()
            )
            family_logits_all.append(
                family_logits.float().cpu().numpy()
            )
            labels_all.extend(labels.tolist())

    action_logits_np = np.concatenate(
        action_logits_all,
        axis=0,
    )
    family_logits_np = np.concatenate(
        family_logits_all,
        axis=0,
    )
    labels_np = np.asarray(
        labels_all,
        dtype=np.int64,
    )

    best_score = -1.0
    best_weight = 0.0
    best_predictions = None

    family_index = np.asarray(
        ACTION_TO_FAMILY,
        dtype=np.int64,
    )

    for family_weight in FAMILY_LOGIT_WEIGHTS:
        final_logits = (
            action_logits_np
            + family_weight
            * family_logits_np[:, family_index]
        )
        predictions = final_logits.argmax(axis=1)

        score = f1_score(
            labels_np,
            predictions,
            labels=list(
                range(len(ALL_CLASSES))
            ),
            average="macro",
            zero_division=0,
        )

        if score > best_score:
            best_score = float(score)
            best_weight = float(family_weight)
            best_predictions = predictions

    return (
        best_score,
        best_weight,
        labels_np,
        best_predictions,
    )


def save_model(
    model: QwenSegmentClassifier,
    tokenizer,
    best_score: float,
    family_logit_weight: float,
) -> None:
    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.backbone.save_pretrained(
        str(SAVE_DIR / "adapter")
    )
    tokenizer.save_pretrained(
        str(SAVE_DIR)
    )

    non_backbone_state = {
        key: value.cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("backbone.")
    }

    torch.save(
        non_backbone_state,
        SAVE_DIR / "heads.pt",
    )

    metadata = {
        "base_model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "classes": ALL_CLASSES,
        "families": FAMILY_NAMES,
        "action_to_family": ACTION_TO_FAMILY,
        "structured_dim": STRUCTURED_DIM,
        "segment_budgets": SEGMENT_BUDGETS,
        "family_logit_weight": family_logit_weight,
        "validation_macro_f1": best_score,
    }

    (SAVE_DIR / "metadata.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU가 필요합니다."
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda")

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )
    print("Load data...")

    samples, targets = load_data()

    if args.overfit_check:
        samples, targets = balanced_subset(
            samples,
            targets,
            per_class=10,
        )
        print(
            "Overfit check samples:",
            len(samples),
        )
    elif args.smoke:
        samples, targets = balanced_subset(
            samples,
            targets,
            per_class=200,
        )
        print("Smoke samples:", len(samples))
    else:
        print("Full samples:", len(samples))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    labels = np.asarray([
        LABEL2ID[target]
        for target in targets
    ], dtype=np.int64)

    if args.overfit_check:
        train_samples = samples
        validation_samples = samples
        train_labels = labels
        validation_labels = labels
    else:
        (
            train_indices,
            validation_indices,
            all_labels,
        ) = make_split(samples, targets)

        train_samples = [
            samples[index]
            for index in train_indices
        ]
        validation_samples = [
            samples[index]
            for index in validation_indices
        ]
        train_labels = all_labels[
            train_indices
        ]
        validation_labels = all_labels[
            validation_indices
        ]

    print(
        f"train={len(train_samples)} "
        f"val={len(validation_samples)}"
    )
    print(
        "Structured dim:",
        STRUCTURED_DIM,
    )

    train_dataset = EncodedSegmentDataset(
        train_samples,
        train_labels,
        tokenizer,
        chunk_size=args.tokenize_chunk_size,
        description="Tokenize train",
    )
    validation_dataset = EncodedSegmentDataset(
        validation_samples,
        validation_labels,
        tokenizer,
        chunk_size=args.tokenize_chunk_size,
        description="Tokenize val",
    )

    collator = SegmentCollator(
        tokenizer.pad_token_id
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=max(
            args.batch_size * 2,
            16,
        ),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    print("Load Qwen backbone...")

    backbone = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    backbone.config.use_cache = False
    backbone.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    backbone = get_peft_model(
        backbone,
        lora_config,
    )

    for parameter in backbone.parameters():
        if parameter.requires_grad:
            parameter.data = (
                parameter.data.float()
            )

    model = QwenSegmentClassifier(
        backbone=backbone,
        hidden_size=backbone.config.hidden_size,
        structured_dim=STRUCTURED_DIM,
        num_labels=len(ALL_CLASSES),
        num_families=len(FAMILY_NAMES),
    ).to(device)

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "Trainable parameters:",
        f"{trainable:,} / {total:,} "
        f"({100 * trainable / total:.3f}%)",
    )

    action_counts = np.bincount(
        train_labels,
        minlength=len(ALL_CLASSES),
    ).astype(np.float32)

    family_labels_np = np.asarray([
        ACTION_TO_FAMILY[int(label)]
        for label in train_labels
    ], dtype=np.int64)

    family_counts = np.bincount(
        family_labels_np,
        minlength=len(FAMILY_NAMES),
    ).astype(np.float32)

    if args.overfit_check:
        action_weights = torch.ones(
            len(ALL_CLASSES),
            dtype=torch.float32,
            device=device,
        )
        family_weights = torch.ones(
            len(FAMILY_NAMES),
            dtype=torch.float32,
            device=device,
        )
    else:
        action_weight_values = np.power(
            len(train_labels)
            / (
                len(ALL_CLASSES)
                * np.maximum(
                    action_counts,
                    1,
                )
            ),
            0.25,
        )
        family_weight_values = np.power(
            len(family_labels_np)
            / (
                len(FAMILY_NAMES)
                * np.maximum(
                    family_counts,
                    1,
                )
            ),
            0.25,
        )

        action_weights = torch.tensor(
            action_weight_values,
            dtype=torch.float32,
            device=device,
        )
        family_weights = torch.tensor(
            family_weight_values,
            dtype=torch.float32,
            device=device,
        )

    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if (
            parameter.requires_grad
            and not name.startswith("backbone.")
        )
    ]
    head_ids = {
        id(parameter)
        for parameter in head_parameters
    }
    lora_parameters = [
        parameter
        for parameter in model.parameters()
        if (
            parameter.requires_grad
            and id(parameter) not in head_ids
        )
    ]

    optimizer = torch.optim.AdamW([
        {
            "params": head_parameters,
            "lr": args.head_lr,
            "weight_decay": 0.01,
        },
        {
            "params": lora_parameters,
            "lr": args.lora_lr,
            "weight_decay": 0.01,
        },
    ])

    updates_per_epoch = max(
        1,
        (
            len(train_loader)
            + args.grad_accum
            - 1
        ) // args.grad_accum,
    )
    total_updates = (
        updates_per_epoch
        * args.epochs
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(
            1,
            int(total_updates * 0.05),
        ),
        num_training_steps=total_updates,
    )

    scaler = torch.amp.GradScaler("cuda")
    best_score = -1.0
    best_family_logit_weight = 0.0
    best_true = None
    best_pred = None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = torch.zeros(
            (),
            dtype=torch.float32,
            device=device,
        )

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch + 1}/"
                f"{args.epochs}"
            ),
        )

        for step, batch in enumerate(
            progress,
            start=1,
        ):
            labels_batch = batch.pop(
                "labels"
            ).to(
                device,
                non_blocking=True,
            )
            family_labels_batch = batch.pop(
                "family_labels"
            ).to(
                device,
                non_blocking=True,
            )

            batch = {
                key: value.to(
                    device,
                    non_blocking=True,
                )
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                (
                    action_logits,
                    family_logits,
                ) = model(**batch)

                action_loss = F.cross_entropy(
                    action_logits,
                    labels_batch,
                    weight=action_weights,
                )
                family_loss = F.cross_entropy(
                    family_logits,
                    family_labels_batch,
                    weight=family_weights,
                )

                raw_loss = (
                    action_loss
                    + args.family_loss_weight
                    * family_loss
                )
                loss = (
                    raw_loss
                    / args.grad_accum
                )

            scaler.scale(loss).backward()
            running_loss += raw_loss.detach()

            if (
                step % args.grad_accum == 0
                or step == len(train_loader)
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(
                    set_to_none=True
                )

            if step % 100 == 0:
                progress.set_postfix(
                    loss=f"{raw_loss.detach().item():.4f}"
                )

        (
            macro_f1,
            family_logit_weight,
            y_true,
            y_pred,
        ) = evaluate(
            model,
            validation_loader,
            device,
        )

        epoch_loss = (
            running_loss
            / len(train_loader)
        ).item()

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={epoch_loss:.6f} "
            f"macro_f1={macro_f1:.6f} "
            f"family_logit_weight="
            f"{family_logit_weight:.2f}"
        )

        if macro_f1 > best_score:
            best_score = macro_f1
            best_family_logit_weight = (
                family_logit_weight
            )
            best_true = y_true
            best_pred = y_pred

            save_model(
                model,
                tokenizer,
                best_score,
                best_family_logit_weight,
            )

    print(
        f"Best Macro-F1: "
        f"{best_score:.6f}"
    )
    print(
        "Best family logit weight:",
        best_family_logit_weight,
    )

    print(
        classification_report(
            best_true,
            best_pred,
            labels=list(
                range(len(ALL_CLASSES))
            ),
            target_names=ALL_CLASSES,
            digits=4,
            zero_division=0,
        )
    )

    print("Saved:", SAVE_DIR)


if __name__ == "__main__":
    main()
