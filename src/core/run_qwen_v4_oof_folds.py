from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

import train_qwen_segment_v4 as v4


SEED = 42
NUM_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially train independent Qwen V4 folds 1-4, "
            "save each held-fold logits, and assemble 70k OOF logits. "
            "Existing Fold 0 logits are reused."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/train.jsonl"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/train_labels.csv"),
    )
    parser.add_argument(
        "--reference-v4-dir",
        type=Path,
        default=Path("model/qwen_segment_v4"),
    )
    parser.add_argument(
        "--fold0-logits",
        type=Path,
        default=Path("validation_logits_v4.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/qwen_v4_oof"),
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="1,2,3,4",
        help="Comma-separated folds to train. Fold 0 is reused.",
    )
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
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain folds even when completed outputs exist.",
    )
    parser.add_argument(
        "--keep-fold-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep adapters and heads for every fold.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(
    data_path: Path,
    labels_path: Path,
) -> Tuple[List[dict], np.ndarray]:
    samples: List[dict] = []

    with data_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    with labels_path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        label_map: Dict[str, str] = {
            str(row["id"]): str(row["action"])
            for row in csv.DictReader(file)
        }

    labels = np.asarray([
        v4.LABEL2ID[
            label_map[str(sample["id"])]
        ]
        for sample in samples
    ], dtype=np.int64)

    return samples, labels


def balanced_subset_indices(
    labels: np.ndarray,
    per_class: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: List[int] = []

    for class_id in range(len(v4.ALL_CLASSES)):
        candidates = np.flatnonzero(
            labels == class_id
        )
        rng.shuffle(candidates)
        selected.extend(
            candidates[:per_class].tolist()
        )

    selected_array = np.asarray(
        selected,
        dtype=np.int64,
    )
    rng.shuffle(selected_array)
    return selected_array


def build_splits(
    samples: Sequence[dict],
    labels: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([
        str(sample["id"]).rsplit(
            "-step_",
            1,
        )[0]
        for sample in samples
    ])

    splitter = StratifiedGroupKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=SEED,
    )
    splits = list(
        splitter.split(
            np.zeros(len(labels)),
            labels,
            groups,
        )
    )

    for fold_index, (
        train_indices,
        validation_indices,
    ) in enumerate(splits):
        overlap = (
            set(groups[train_indices])
            & set(groups[validation_indices])
        )
        if overlap:
            raise RuntimeError(
                f"Fold {fold_index}: session overlap "
                f"{len(overlap)}"
            )

    return splits


def parse_fold_list(value: str) -> List[int]:
    folds: List[int] = []

    for part in value.split(","):
        part = part.strip()
        if not part:
            continue

        fold = int(part)
        if fold < 1 or fold >= NUM_FOLDS:
            raise ValueError(
                "Only folds 1,2,3,4 can be trained. "
                "Fold 0 is reused from --fold0-logits."
            )
        folds.append(fold)

    if not folds:
        raise ValueError("No folds requested.")

    return sorted(set(folds))


def load_reference_settings(
    reference_v4_dir: Path,
) -> Tuple[str, dict]:
    metadata_path = (
        reference_v4_dir / "metadata.json"
    )
    postprocess_path = (
        reference_v4_dir / "postprocess.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not postprocess_path.exists():
        raise FileNotFoundError(postprocess_path)

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )
    postprocess = json.loads(
        postprocess_path.read_text(
            encoding="utf-8"
        )
    )

    return str(metadata["base_model"]), postprocess


def class_weight_values(
    labels: np.ndarray,
    class_count: int,
) -> np.ndarray:
    counts = np.bincount(
        labels,
        minlength=class_count,
    ).astype(np.float32)

    return np.power(
        len(labels)
        / (
            class_count
            * np.maximum(counts, 1.0)
        ),
        0.25,
    ).astype(np.float32)


def apply_postprocess(
    action_logits: np.ndarray,
    family_logits: np.ndarray,
    training_class_weights: np.ndarray,
    postprocess: dict,
) -> np.ndarray:
    family_index = np.asarray(
        v4.ACTION_TO_FAMILY,
        dtype=np.int64,
    )

    return (
        action_logits.astype(np.float64)
        / float(
            postprocess["action_temperature"]
        )
        + float(
            postprocess["family_weight"]
        )
        * family_logits.astype(np.float64)[
            :,
            family_index,
        ]
        - float(
            postprocess["prior_beta"]
        )
        * np.log(
            np.maximum(
                training_class_weights.astype(
                    np.float64
                ),
                1e-12,
            )
        )[None, :]
    )


def macro_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> float:
    return float(
        f1_score(
            labels,
            predictions,
            labels=np.arange(
                len(v4.ALL_CLASSES)
            ),
            average="macro",
            zero_division=0,
        )
    )


def evaluate_raw_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    model.eval()

    action_parts: List[np.ndarray] = []
    family_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc="Validation logits",
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
                (
                    action_logits,
                    family_logits,
                ) = model(**batch)

            action_parts.append(
                action_logits.float().cpu().numpy()
            )
            family_parts.append(
                family_logits.float().cpu().numpy()
            )
            label_parts.append(
                labels.numpy()
            )

    return (
        np.concatenate(
            action_parts,
            axis=0,
        ).astype(np.float32),
        np.concatenate(
            family_parts,
            axis=0,
        ).astype(np.float32),
        np.concatenate(
            label_parts,
            axis=0,
        ).astype(np.int64),
    )


def build_fresh_model(
    model_name: str,
    tokenizer,
    device: torch.device,
) -> v4.QwenSegmentClassifier:
    backbone = AutoModel.from_pretrained(
        model_name,
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

    model = v4.QwenSegmentClassifier(
        backbone=backbone,
        hidden_size=backbone.config.hidden_size,
        structured_dim=v4.STRUCTURED_DIM,
        num_labels=len(v4.ALL_CLASSES),
        num_families=len(v4.FAMILY_NAMES),
    ).to(device)

    return model


def save_fold_model(
    model: v4.QwenSegmentClassifier,
    tokenizer,
    fold_dir: Path,
    metadata: dict,
    postprocess: dict,
) -> None:
    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    adapter_dir = fold_dir / "adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)

    model.backbone.save_pretrained(
        str(adapter_dir)
    )
    tokenizer.save_pretrained(
        str(fold_dir)
    )

    head_state = {
        key: value.detach().cpu()
        for key, value
        in model.state_dict().items()
        if not key.startswith(
            "backbone."
        )
    }
    torch.save(
        head_state,
        fold_dir / "heads.pt",
    )

    (
        fold_dir / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (
        fold_dir / "postprocess.json"
    ).write_text(
        json.dumps(
            postprocess,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def fold_is_complete(
    fold_dir: Path,
) -> bool:
    required = [
        fold_dir / "oof_logits.npz",
        fold_dir / "status.json",
    ]
    if not all(
        path.exists()
        for path in required
    ):
        return False

    try:
        status = json.loads(
            (
                fold_dir / "status.json"
            ).read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return False

    return status.get("status") == "complete"


def train_one_fold(
    fold_index: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    full_dataset: v4.EncodedSegmentDataset,
    labels: np.ndarray,
    tokenizer,
    model_name: str,
    reference_postprocess: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    fold_started = time.perf_counter()
    fold_dir = (
        args.output_dir
        / f"fold_{fold_index}"
    )
    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        fold_is_complete(fold_dir)
        and not args.force
    ):
        print(
            f"Fold {fold_index}: already complete, skip."
        )
        return json.loads(
            (
                fold_dir / "status.json"
            ).read_text(
                encoding="utf-8"
            )
        )

    if args.force and fold_dir.exists():
        shutil.rmtree(fold_dir)
        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    fold_seed = SEED + fold_index * 1000
    set_seed(fold_seed)

    train_labels = labels[
        train_indices
    ]
    validation_labels = labels[
        validation_indices
    ]

    action_weight_values = (
        class_weight_values(
            train_labels,
            len(v4.ALL_CLASSES),
        )
    )
    family_train_labels = np.asarray([
        v4.ACTION_TO_FAMILY[
            int(label)
        ]
        for label in train_labels
    ], dtype=np.int64)
    family_weight_values = (
        class_weight_values(
            family_train_labels,
            len(v4.FAMILY_NAMES),
        )
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

    train_loader = DataLoader(
        Subset(
            full_dataset,
            train_indices.tolist(),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=v4.SegmentCollator(
            tokenizer.pad_token_id
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        Subset(
            full_dataset,
            validation_indices.tolist(),
        ),
        batch_size=max(
            args.validation_batch_size,
            args.batch_size * 2,
        ),
        shuffle=False,
        collate_fn=v4.SegmentCollator(
            tokenizer.pad_token_id
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print()
    print("=" * 72)
    print(
        f"Train Fold {fold_index}: "
        f"train={len(train_indices)} "
        f"validation={len(validation_indices)}"
    )
    print("=" * 72)

    model = build_fresh_model(
        model_name,
        tokenizer,
        device,
    )

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
        f"{trainable:,}/{total:,} "
        f"({100.0 * trainable / total:.3f}%)",
    )

    head_parameters = [
        parameter
        for name, parameter
        in model.named_parameters()
        if (
            parameter.requires_grad
            and not name.startswith(
                "backbone."
            )
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
            and id(parameter)
            not in head_ids
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
        )
        // args.grad_accum,
    )
    total_updates = (
        updates_per_epoch
        * args.epochs
    )

    scheduler = (
        get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(
                1,
                int(total_updates * 0.05),
            ),
            num_training_steps=(
                total_updates
            ),
        )
    )
    scaler = torch.amp.GradScaler(
        "cuda"
    )

    best_score = -1.0
    best_epoch = 0
    best_action_logits = None
    best_family_logits = None
    best_validation_labels = None
    best_predictions = None
    epoch_records: List[dict] = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()
        optimizer.zero_grad(
            set_to_none=True
        )
        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=(
                f"Fold {fold_index} "
                f"Epoch {epoch}/{args.epochs}"
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
            family_labels_batch = (
                batch.pop(
                    "family_labels"
                ).to(
                    device,
                    non_blocking=True,
                )
            )
            batch = {
                key: value.to(
                    device,
                    non_blocking=True,
                )
                for key, value
                in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                (
                    action_logits,
                    family_logits,
                ) = model(**batch)

                action_loss = (
                    F.cross_entropy(
                        action_logits,
                        labels_batch,
                        weight=action_weights,
                    )
                )
                family_loss = (
                    F.cross_entropy(
                        family_logits,
                        family_labels_batch,
                        weight=family_weights,
                    )
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
            running_loss += float(
                raw_loss.detach().item()
            )

            should_update = (
                step % args.grad_accum == 0
                or step == len(train_loader)
            )

            if should_update:
                scaler.unscale_(
                    optimizer
                )
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
                    loss=(
                        f"{raw_loss.detach().item():.4f}"
                    )
                )

        (
            action_logits_np,
            family_logits_np,
            labels_np,
        ) = evaluate_raw_logits(
            model,
            validation_loader,
            device,
        )

        if not np.array_equal(
            labels_np,
            validation_labels,
        ):
            raise RuntimeError(
                f"Fold {fold_index}: validation "
                "label order mismatch."
            )

        final_logits = apply_postprocess(
            action_logits_np,
            family_logits_np,
            action_weight_values,
            reference_postprocess,
        )
        predictions = (
            final_logits.argmax(
                axis=1
            )
        )
        score = macro_f1(
            labels_np,
            predictions,
        )
        epoch_loss = (
            running_loss
            / max(
                1,
                len(train_loader),
            )
        )

        epoch_record = {
            "epoch": epoch,
            "training_loss": (
                epoch_loss
            ),
            "validation_macro_f1": (
                score
            ),
        }
        epoch_records.append(
            epoch_record
        )

        print(
            f"Fold {fold_index} "
            f"Epoch {epoch}/{args.epochs} "
            f"loss={epoch_loss:.6f} "
            f"macro_f1={score:.6f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_action_logits = (
                action_logits_np.copy()
            )
            best_family_logits = (
                family_logits_np.copy()
            )
            best_validation_labels = (
                labels_np.copy()
            )
            best_predictions = (
                predictions.copy()
            )

            fold_postprocess = {
                **reference_postprocess,
                "training_class_weights": [
                    float(value)
                    for value
                    in action_weight_values
                ],
                "source": (
                    "Fixed V4 postprocess parameters; "
                    "class weights recomputed for "
                    f"OOF fold {fold_index}."
                ),
            }
            metadata = {
                "base_model": model_name,
                "architecture": (
                    "qwen_segment_v4"
                ),
                "fold_index": (
                    fold_index
                ),
                "num_folds": NUM_FOLDS,
                "split_seed": SEED,
                "training_seed": (
                    fold_seed
                ),
                "train_samples": int(
                    len(train_indices)
                ),
                "validation_samples": int(
                    len(
                        validation_indices
                    )
                ),
                "best_epoch": (
                    best_epoch
                ),
                "validation_macro_f1": (
                    best_score
                ),
                "epochs_requested": (
                    args.epochs
                ),
                "batch_size": (
                    args.batch_size
                ),
                "grad_accum": (
                    args.grad_accum
                ),
                "head_lr": (
                    args.head_lr
                ),
                "lora_lr": (
                    args.lora_lr
                ),
                "family_loss_weight": (
                    args.family_loss_weight
                ),
                "classes": (
                    v4.ALL_CLASSES
                ),
                "families": (
                    v4.FAMILY_NAMES
                ),
                "action_to_family": (
                    v4.ACTION_TO_FAMILY
                ),
                "structured_dim": (
                    v4.STRUCTURED_DIM
                ),
                "segment_budgets": (
                    v4.SEGMENT_BUDGETS
                ),
            }

            if args.keep_fold_models:
                save_fold_model(
                    model,
                    tokenizer,
                    fold_dir,
                    metadata,
                    fold_postprocess,
                )
            else:
                (
                    fold_dir
                    / "metadata.json"
                ).write_text(
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                (
                    fold_dir
                    / "postprocess.json"
                ).write_text(
                    json.dumps(
                        fold_postprocess,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    if (
        best_action_logits is None
        or best_family_logits is None
        or best_validation_labels is None
        or best_predictions is None
    ):
        raise RuntimeError(
            f"Fold {fold_index}: "
            "no best checkpoint."
        )

    best_final_logits = apply_postprocess(
        best_action_logits,
        best_family_logits,
        action_weight_values,
        reference_postprocess,
    ).astype(np.float32)

    np.savez_compressed(
        fold_dir / "oof_logits.npz",
        action_logits=(
            best_action_logits.astype(
                np.float32
            )
        ),
        family_logits=(
            best_family_logits.astype(
                np.float32
            )
        ),
        final_logits=(
            best_final_logits
        ),
        labels=(
            best_validation_labels
        ),
        validation_indices=(
            validation_indices.astype(
                np.int64
            )
        ),
        fold_index=np.asarray(
            [fold_index],
            dtype=np.int64,
        ),
        best_epoch=np.asarray(
            [best_epoch],
            dtype=np.int64,
        ),
    )

    report = classification_report(
        best_validation_labels,
        best_predictions,
        labels=np.arange(
            len(v4.ALL_CLASSES)
        ),
        target_names=v4.ALL_CLASSES,
        digits=6,
        zero_division=0,
    )
    (
        fold_dir
        / "classification_report.txt"
    ).write_text(
        report,
        encoding="utf-8",
    )

    status = {
        "status": "complete",
        "fold_index": fold_index,
        "best_epoch": best_epoch,
        "best_macro_f1": (
            best_score
        ),
        "train_samples": int(
            len(train_indices)
        ),
        "validation_samples": int(
            len(validation_indices)
        ),
        "elapsed_seconds": float(
            time.perf_counter()
            - fold_started
        ),
        "epoch_records": (
            epoch_records
        ),
        "kept_model": bool(
            args.keep_fold_models
        ),
    }
    (
        fold_dir / "status.json"
    ).write_text(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Fold {fold_index} complete: "
        f"best_epoch={best_epoch} "
        f"macro_f1={best_score:.6f}"
    )
    print(
        "Saved:",
        fold_dir,
    )

    del model
    del optimizer
    del scheduler
    del scaler
    del train_loader
    del validation_loader
    gc.collect()
    torch.cuda.empty_cache()

    return status


def validate_fold0(
    fold0_logits_path: Path,
    expected_indices: np.ndarray,
    expected_labels: np.ndarray,
) -> dict:
    if not fold0_logits_path.exists():
        raise FileNotFoundError(
            fold0_logits_path
        )

    payload = np.load(
        fold0_logits_path
    )
    required_keys = {
        "action_logits",
        "family_logits",
        "labels",
        "validation_indices",
    }
    missing = (
        required_keys
        - set(payload.files)
    )
    if missing:
        raise RuntimeError(
            "Fold 0 logits missing keys: "
            f"{sorted(missing)}"
        )

    actual_indices = payload[
        "validation_indices"
    ].astype(np.int64)
    actual_labels = payload[
        "labels"
    ].astype(np.int64)

    if not np.array_equal(
        actual_indices,
        expected_indices,
    ):
        mismatch = int(
            np.sum(
                actual_indices
                != expected_indices
            )
        ) if len(actual_indices) == len(
            expected_indices
        ) else -1
        raise RuntimeError(
            "Existing Fold 0 indices do not match "
            "the current StratifiedGroupKFold split. "
            f"Mismatch={mismatch}"
        )

    if not np.array_equal(
        actual_labels,
        expected_labels,
    ):
        raise RuntimeError(
            "Existing Fold 0 labels do not match."
        )

    return {
        "samples": int(
            len(actual_indices)
        ),
        "path": str(
            fold0_logits_path
        ),
    }


def assemble_oof(
    output_dir: Path,
    fold0_logits_path: Path,
    reference_postprocess: dict,
    splits: Sequence[
        Tuple[np.ndarray, np.ndarray]
    ],
    labels: np.ndarray,
) -> dict:
    total_samples = len(labels)
    action_logits_all = np.full(
        (
            total_samples,
            len(v4.ALL_CLASSES),
        ),
        np.nan,
        dtype=np.float32,
    )
    family_logits_all = np.full(
        (
            total_samples,
            len(v4.FAMILY_NAMES),
        ),
        np.nan,
        dtype=np.float32,
    )
    final_logits_all = np.full(
        (
            total_samples,
            len(v4.ALL_CLASSES),
        ),
        np.nan,
        dtype=np.float32,
    )
    fold_ids = np.full(
        total_samples,
        -1,
        dtype=np.int8,
    )
    best_epochs = np.full(
        NUM_FOLDS,
        -1,
        dtype=np.int8,
    )

    fold_scores: Dict[str, float] = {}

    for fold_index in range(
        NUM_FOLDS
    ):
        if fold_index == 0:
            path = fold0_logits_path
        else:
            path = (
                output_dir
                / f"fold_{fold_index}"
                / "oof_logits.npz"
            )

        if not path.exists():
            return {
                "assembled": False,
                "reason": (
                    f"Missing fold {fold_index}: "
                    f"{path}"
                ),
            }

        payload = np.load(path)
        indices = payload[
            "validation_indices"
        ].astype(np.int64)
        labels_local = payload[
            "labels"
        ].astype(np.int64)
        action_logits = payload[
            "action_logits"
        ].astype(np.float32)
        family_logits = payload[
            "family_logits"
        ].astype(np.float32)

        expected_indices = splits[
            fold_index
        ][1]
        if not np.array_equal(
            indices,
            expected_indices,
        ):
            raise RuntimeError(
                f"Fold {fold_index} index mismatch "
                "during OOF assembly."
            )
        if not np.array_equal(
            labels_local,
            labels[indices],
        ):
            raise RuntimeError(
                f"Fold {fold_index} label mismatch "
                "during OOF assembly."
            )

        if "final_logits" in payload.files:
            final_logits = payload[
                "final_logits"
            ].astype(np.float32)
        else:
            if fold_index != 0:
                raise RuntimeError(
                    f"Fold {fold_index} has no final_logits."
                )

            reference_weights = np.asarray(
                reference_postprocess[
                    "training_class_weights"
                ],
                dtype=np.float32,
            )
            final_logits = apply_postprocess(
                action_logits,
                family_logits,
                reference_weights,
                reference_postprocess,
            ).astype(np.float32)

        action_logits_all[
            indices
        ] = action_logits
        family_logits_all[
            indices
        ] = family_logits
        final_logits_all[
            indices
        ] = final_logits
        fold_ids[
            indices
        ] = fold_index

        if "best_epoch" in payload.files:
            best_epochs[
                fold_index
            ] = int(
                payload[
                    "best_epoch"
                ][0]
            )

        predictions = final_logits.argmax(
            axis=1
        )
        fold_scores[
            str(fold_index)
        ] = macro_f1(
            labels_local,
            predictions,
        )

    if np.isnan(
        action_logits_all
    ).any():
        raise RuntimeError(
            "OOF action logits contain missing values."
        )
    if np.isnan(
        family_logits_all
    ).any():
        raise RuntimeError(
            "OOF family logits contain missing values."
        )
    if np.isnan(
        final_logits_all
    ).any():
        raise RuntimeError(
            "OOF final logits contain missing values."
        )
    if np.any(
        fold_ids < 0
    ):
        raise RuntimeError(
            "OOF fold ids contain missing values."
        )

    predictions = final_logits_all.argmax(
        axis=1
    )
    overall_score = macro_f1(
        labels,
        predictions,
    )

    np.savez_compressed(
        output_dir
        / "oof_logits_all_70000.npz",
        action_logits=(
            action_logits_all
        ),
        family_logits=(
            family_logits_all
        ),
        final_logits=(
            final_logits_all
        ),
        labels=labels.astype(
            np.int64
        ),
        fold_ids=fold_ids,
        best_epochs=best_epochs,
    )

    report = classification_report(
        labels,
        predictions,
        labels=np.arange(
            len(v4.ALL_CLASSES)
        ),
        target_names=v4.ALL_CLASSES,
        digits=6,
        zero_division=0,
    )
    (
        output_dir
        / "oof_classification_report.txt"
    ).write_text(
        report,
        encoding="utf-8",
    )

    result = {
        "assembled": True,
        "samples": int(
            total_samples
        ),
        "overall_oof_macro_f1": (
            overall_score
        ),
        "fold_macro_f1": (
            fold_scores
        ),
        "output": str(
            output_dir
            / "oof_logits_all_70000.npz"
        ),
    }
    (
        output_dir
        / "oof_summary.json"
    ).write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(
        "70,000-sample OOF assembly complete"
    )
    print(
        f"Overall OOF Macro-F1: "
        f"{overall_score:.6f}"
    )
    print(
        "Fold scores:",
        json.dumps(
            fold_scores,
            ensure_ascii=False,
        ),
    )
    print(
        "Saved:",
        output_dir
        / "oof_logits_all_70000.npz",
    )
    print("=" * 72)

    return result


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    requested_folds = (
        parse_fold_list(
            args.folds
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision(
        "high"
    )
    device = torch.device("cuda")

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )
    print(
        "Requested folds:",
        requested_folds,
    )
    print("Load data...")

    samples, labels = load_data(
        args.data,
        args.labels,
    )

    if args.smoke:
        subset_indices = (
            balanced_subset_indices(
                labels,
                per_class=60,
                seed=SEED,
            )
        )
        samples = [
            samples[int(index)]
            for index in subset_indices
        ]
        labels = labels[
            subset_indices
        ]
        args.epochs = min(
            args.epochs,
            1,
        )
        requested_folds = [
            requested_folds[0]
        ]
        print(
            "Smoke samples:",
            len(samples),
        )
        print(
            "Smoke fold:",
            requested_folds[0],
        )
        print(
            "NOTE: smoke mode cannot reuse or "
            "assemble the real Fold 0 logits."
        )
    else:
        print(
            "Full samples:",
            len(samples),
        )

    splits = build_splits(
        samples,
        labels,
    )

    model_name, reference_postprocess = (
        load_reference_settings(
            args.reference_v4_dir
        )
    )

    if not args.smoke:
        fold0_info = validate_fold0(
            args.fold0_logits,
            splits[0][1],
            labels[
                splits[0][1]
            ],
        )
        print(
            "Fold 0 logits verified:",
            fold0_info,
        )

    print("Load tokenizer...")
    tokenizer = (
        AutoTokenizer.from_pretrained(
            str(
                args.reference_v4_dir
            ),
            use_fast=True,
            local_files_only=True,
        )
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        "Tokenize all samples once..."
    )
    full_dataset = (
        v4.EncodedSegmentDataset(
            samples,
            labels,
            tokenizer,
            chunk_size=(
                args.tokenize_chunk_size
            ),
            description=(
                "Tokenize all OOF samples"
            ),
        )
    )

    run_manifest = {
        "status": "running",
        "requested_folds": (
            requested_folds
        ),
        "model_name": model_name,
        "split_seed": SEED,
        "samples": len(samples),
        "started_at_unix": (
            time.time()
        ),
        "folds": {},
    }
    manifest_path = (
        args.output_dir
        / "run_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            run_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for fold_index in requested_folds:
        train_indices, validation_indices = (
            splits[fold_index]
        )

        status = train_one_fold(
            fold_index=fold_index,
            train_indices=train_indices,
            validation_indices=(
                validation_indices
            ),
            full_dataset=full_dataset,
            labels=labels,
            tokenizer=tokenizer,
            model_name=model_name,
            reference_postprocess=(
                reference_postprocess
            ),
            args=args,
            device=device,
        )

        run_manifest[
            "folds"
        ][str(fold_index)] = status
        manifest_path.write_text(
            json.dumps(
                run_manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if args.smoke:
        assembly = {
            "assembled": False,
            "reason": (
                "Smoke mode does not assemble "
                "the real 70k OOF file."
            ),
        }
    else:
        assembly = assemble_oof(
            output_dir=(
                args.output_dir
            ),
            fold0_logits_path=(
                args.fold0_logits
            ),
            reference_postprocess=(
                reference_postprocess
            ),
            splits=splits,
            labels=labels,
        )

    run_manifest["status"] = (
        "complete"
        if (
            args.smoke
            or assembly.get(
                "assembled",
                False,
            )
        )
        else "partial"
    )
    run_manifest["assembly"] = (
        assembly
    )
    run_manifest[
        "finished_at_unix"
    ] = time.time()
    manifest_path.write_text(
        json.dumps(
            run_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Sequential OOF run finished.")
    print(
        "Manifest:",
        manifest_path,
    )
    if not args.smoke:
        if assembly.get(
            "assembled",
            False,
        ):
            print(
                "Next input file:",
                assembly["output"],
            )
        else:
            print(
                "OOF assembly is partial:",
                assembly["reason"],
            )


if __name__ == "__main__":
    main()
