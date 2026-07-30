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
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model

import train_qwen_distill_v27_tree_teacher as base
from feature_utils_qwen_v4 import ACTION_TO_FAMILY, ALL_CLASSES, FAMILY_NAMES, STRUCTURED_DIM

SEED = 42
NUM_CLASSES = len(ALL_CLASSES)
LABEL2ID = {label: i for i, label in enumerate(ALL_CLASSES)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
FAMILY_INDEX = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)


DEFAULT_CONFUSION_PAIRS = "read_file:grep_search,read_file:list_directory,grep_search:glob_pattern,ask_user:plan_task,run_bash:run_tests,run_tests:lint_or_typecheck"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V28 hard-negative / confusion-pair retraining for the V12 Qwen 0.5B checkpoint. "
            "It upweights OOF-mispredicted hard examples while downweighting distillation "
            "on rows where the OOF teacher is wrong, so the old teacher does not fight the hard label."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--init-dir", "--v4-dir", dest="init_dir", type=Path, default=Path("model/qwen_distill_v12_eval"),
                        help="Checkpoint directory containing adapter/, heads.pt, metadata.json, postprocess.json.")
    parser.add_argument("--oof-logits", type=Path, default=Path("model/qwen_v4_oof/oof_logits_all_70000.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/qwen_distill_v28_hard_negative_eval"))
    parser.add_argument("--mode", choices=["eval", "full"], default="eval")
    parser.add_argument("--eval-fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--lora-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--tokenize-chunk-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--history-budget", type=int, default=0)
    parser.add_argument("--action-budget", type=int, default=0)
    parser.add_argument("--meta-budget", type=int, default=0)
    parser.add_argument("--current-budget", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")

    # Base V12-style objective. Defaults are slightly more label-centric than V12/V27.
    parser.add_argument("--hard-final-weight", type=float, default=0.85)
    parser.add_argument("--hard-action-weight", type=float, default=0.45)
    parser.add_argument("--family-weight", type=float, default=0.25)
    parser.add_argument("--distill-final-weight", type=float, default=0.35)
    parser.add_argument("--distill-action-weight", type=float, default=0.10)
    parser.add_argument("--distill-family-weight", type=float, default=0.08)
    parser.add_argument("--teacher-ce-weight", type=float, default=0.05)
    parser.add_argument("--distill-temperature", type=float, default=2.0)

    # Hard-negative controls.
    parser.add_argument("--confusion-pairs", type=str, default=DEFAULT_CONFUSION_PAIRS)
    parser.add_argument("--hard-negative-multiplier", type=float, default=2.0,
                        help="Label-loss multiplier for all OOF teacher wrong rows.")
    parser.add_argument("--confusion-pair-multiplier", type=float, default=1.6,
                        help="Additional multiplier for selected true/pred confusion pairs.")
    parser.add_argument("--low-margin-threshold", type=float, default=0.12,
                        help="OOF top1-top2 probability margin threshold for ambiguous rows.")
    parser.add_argument("--low-margin-multiplier", type=float, default=1.25,
                        help="Label-loss multiplier for low-margin rows where the teacher is correct.")
    parser.add_argument("--sample-weight-cap", type=float, default=4.0)
    parser.add_argument("--sample-weight-min", type=float, default=0.45)
    parser.add_argument("--no-class-normalize-hard-weights", action="store_true",
                        help="By default, hard weights are normalized to mean 1 within each true class.")
    parser.add_argument("--wrong-teacher-distill-scale", type=float, default=0.20,
                        help="Distillation multiplier when OOF teacher prediction is wrong.")
    parser.add_argument("--pair-teacher-distill-scale", type=float, default=0.10,
                        help="Maximum distillation multiplier for explicit confusion-pair errors.")

    parser.add_argument("--freeze-lora", action="store_true")
    parser.add_argument("--cold-start", action="store_true",
                        help="Build a fresh LoRA adapter + fresh heads from the base backbone instead of "
                             "resuming from --init-dir's adapter/heads.pt. Use this to isolate the "
                             "hard-negative effect from the 'continuing an already-converged checkpoint on "
                             "the same fold0 train split tends to overfit further' confound -- compare a "
                             "cold-start control (all multipliers=1.0) against cold-start hard-negative at "
                             "the SAME epoch count, rather than comparing either against a resumed run.")
    parser.add_argument("--cold-start-lora-r", type=int, default=16)
    parser.add_argument("--cold-start-lora-alpha", type=int, default=32)
    parser.add_argument("--cold-start-lora-dropout", type=float, default=0.05)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-per-class", type=int, default=100)
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
    groups = np.asarray([str(sample["id"]).rsplit("-step_", 1)[0] for sample in samples])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
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


def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(f1_score(labels, predictions, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0))


def class_f1_values(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return f1_score(labels, predictions, labels=np.arange(NUM_CLASSES), average=None, zero_division=0)


def parse_confusion_pairs(text: str) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    if not text.strip():
        return pairs
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"Bad confusion pair: {raw}. Expected true_label:pred_label")
        left, right = [part.strip() for part in raw.split(":", 1)]
        if left not in LABEL2ID or right not in LABEL2ID:
            raise ValueError(f"Unknown label in confusion pair: {raw}")
        a, b = LABEL2ID[left], LABEL2ID[right]
        pairs.add((a, b))
        pairs.add((b, a))
    return pairs


def build_hard_negative_weights(
    labels: np.ndarray,
    teacher_final_logits: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    teacher_prob = stable_softmax(teacher_final_logits.astype(np.float64))
    teacher_pred = teacher_prob.argmax(axis=1).astype(np.int64)
    top2 = np.partition(teacher_prob, kth=-2, axis=1)[:, -2:]
    margins = top2[:, 1] - top2[:, 0]

    wrong = teacher_pred != labels
    low_margin = margins <= float(args.low_margin_threshold)
    pair_ids = parse_confusion_pairs(args.confusion_pairs)
    pair_error = np.zeros(len(labels), dtype=bool)
    if pair_ids:
        pair_error = np.asarray([(int(y), int(p)) in pair_ids for y, p in zip(labels, teacher_pred)], dtype=bool) & wrong

    sample_weights = np.ones(len(labels), dtype=np.float32)
    sample_weights[wrong] *= float(args.hard_negative_multiplier)
    sample_weights[pair_error] *= float(args.confusion_pair_multiplier)
    sample_weights[(~wrong) & low_margin] *= float(args.low_margin_multiplier)
    sample_weights = np.clip(sample_weights, float(args.sample_weight_min), float(args.sample_weight_cap))

    if not args.no_class_normalize_hard_weights:
        for c in range(NUM_CLASSES):
            mask = labels == c
            if mask.any():
                mean = float(sample_weights[mask].mean())
                if mean > 0:
                    sample_weights[mask] /= mean
        sample_weights = np.clip(sample_weights, float(args.sample_weight_min), float(args.sample_weight_cap))

    distill_weights = np.ones(len(labels), dtype=np.float32)
    distill_weights[wrong] = np.minimum(distill_weights[wrong], float(args.wrong_teacher_distill_scale))
    distill_weights[pair_error] = np.minimum(distill_weights[pair_error], float(args.pair_teacher_distill_scale))
    distill_weights = np.clip(distill_weights, 0.0, 1.0)

    per_class = []
    for c, name in enumerate(ALL_CLASSES):
        mask = labels == c
        if not mask.any():
            continue
        per_class.append({
            "class": name,
            "rows": int(mask.sum()),
            "teacher_wrong": int((wrong & mask).sum()),
            "pair_error": int((pair_error & mask).sum()),
            "low_margin_correct": int(((~wrong) & low_margin & mask).sum()),
            "sample_weight_mean": float(sample_weights[mask].mean()),
            "sample_weight_max": float(sample_weights[mask].max()),
            "distill_weight_mean": float(distill_weights[mask].mean()),
        })

    summary = {
        "teacher_macro_f1": macro_f1(labels, teacher_pred),
        "teacher_wrong": int(wrong.sum()),
        "teacher_wrong_rate": float(wrong.mean()),
        "pair_error": int(pair_error.sum()),
        "low_margin": int(low_margin.sum()),
        "low_margin_correct": int(((~wrong) & low_margin).sum()),
        "sample_weight_mean": float(sample_weights.mean()),
        "sample_weight_min": float(sample_weights.min()),
        "sample_weight_max": float(sample_weights.max()),
        "distill_weight_mean": float(distill_weights.mean()),
        "distill_weight_min": float(distill_weights.min()),
        "distill_weight_max": float(distill_weights.max()),
        "confusion_pairs": args.confusion_pairs,
        "per_class": per_class,
    }
    return sample_weights.astype(np.float32), distill_weights.astype(np.float32), summary


class HardNegativeDistillDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        teacher_action_logits: np.ndarray,
        teacher_family_logits: np.ndarray,
        teacher_final_logits: np.ndarray,
        sample_weights: np.ndarray,
        distill_weights: np.ndarray,
    ):
        self.base_dataset = base_dataset
        self.teacher_action_logits = teacher_action_logits.astype(np.float32)
        self.teacher_family_logits = teacher_family_logits.astype(np.float32)
        self.teacher_final_logits = teacher_final_logits.astype(np.float32)
        self.sample_weights = sample_weights.astype(np.float32)
        self.distill_weights = distill_weights.astype(np.float32)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.base_dataset[index])
        item["teacher_action_logits"] = torch.tensor(self.teacher_action_logits[index], dtype=torch.float32)
        item["teacher_family_logits"] = torch.tensor(self.teacher_family_logits[index], dtype=torch.float32)
        item["teacher_final_logits"] = torch.tensor(self.teacher_final_logits[index], dtype=torch.float32)
        item["sample_weight"] = torch.tensor(float(self.sample_weights[index]), dtype=torch.float32)
        item["distill_weight"] = torch.tensor(float(self.distill_weights[index]), dtype=torch.float32)
        return item


class HardNegativeCollator:
    def __init__(self, pad_token_id: int):
        self.base_collator = base.LongSegmentCollator(pad_token_id)

    def __call__(self, features: List[dict]) -> dict:
        teacher_action = torch.stack([f.pop("teacher_action_logits") for f in features])
        teacher_family = torch.stack([f.pop("teacher_family_logits") for f in features])
        teacher_final = torch.stack([f.pop("teacher_final_logits") for f in features])
        sample_weight = torch.stack([f.pop("sample_weight") for f in features])
        distill_weight = torch.stack([f.pop("distill_weight") for f in features])
        batch = self.base_collator(features)
        batch["teacher_action_logits"] = teacher_action
        batch["teacher_family_logits"] = teacher_family
        batch["teacher_final_logits"] = teacher_final
        batch["sample_weight"] = sample_weight
        batch["distill_weight"] = distill_weight
        return batch


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.float().clamp(min=0.0)
    return (values.float() * weights).sum() / weights.sum().clamp(min=1e-6)


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.cross_entropy(logits.float(), targets, weight=class_weights, reduction="none")
    return weighted_mean(loss, sample_weights)


def weighted_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student = student_logits.float() / float(temperature)
    teacher = teacher_logits.float() / float(temperature)
    teacher_prob = F.softmax(teacher, dim=-1)
    teacher_log_prob = F.log_softmax(teacher, dim=-1)
    student_log_prob = F.log_softmax(student, dim=-1)
    per_sample = (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=-1) * (float(temperature) ** 2)
    return weighted_mean(per_sample, weights)


def save_model(
    model: base.QwenSegmentClassifier,
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

    heads = {key: value.detach().cpu() for key, value in model.state_dict().items() if not key.startswith("backbone.")}
    torch.save(heads, output_dir / "heads.pt")

    metadata_to_save = dict(metadata)
    if evaluation is not None:
        metadata_to_save["validation_macro_f1"] = float(evaluation["macro_f1"])
    (output_dir / "metadata.json").write_text(json.dumps(metadata_to_save, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "postprocess.json").write_text(json.dumps(postprocess, ensure_ascii=False, indent=2), encoding="utf-8")

    if evaluation is not None:
        np.savez_compressed(
            output_dir / "validation_logits_v28.npz",
            action_logits=evaluation["action_logits"],
            family_logits=evaluation["family_logits"],
            labels=evaluation["labels"],
            predictions=evaluation["predictions"],
        )


def make_hard_dataset(
    samples: Sequence[dict],
    labels: np.ndarray,
    indices: np.ndarray,
    tokenizer,
    teacher_action_logits: np.ndarray,
    teacher_family_logits: np.ndarray,
    teacher_final_logits: np.ndarray,
    sample_weights: np.ndarray,
    distill_weights: np.ndarray,
    chunk_size: int,
    description: str,
    max_length: int,
    segment_budgets: Dict[str, int],
) -> HardNegativeDistillDataset:
    subset_samples = [samples[int(i)] for i in indices]
    subset_labels = labels[indices]
    base_dataset = base.LongEncodedSegmentDataset(
        subset_samples,
        subset_labels,
        tokenizer,
        chunk_size=chunk_size,
        description=description,
        max_length=max_length,
        segment_budgets=segment_budgets,
    )
    return HardNegativeDistillDataset(
        base_dataset,
        teacher_action_logits[indices],
        teacher_family_logits[indices],
        teacher_final_logits[indices],
        sample_weights[indices],
        distill_weights[indices],
    )


def make_plain_dataset(
    samples: Sequence[dict],
    labels: np.ndarray,
    indices: np.ndarray,
    tokenizer,
    chunk_size: int,
    description: str,
    max_length: int,
    segment_budgets: Dict[str, int],
):
    subset_samples = [samples[int(i)] for i in indices]
    subset_labels = labels[indices]
    return base.LongEncodedSegmentDataset(
        subset_samples,
        subset_labels,
        tokenizer,
        chunk_size=chunk_size,
        description=description,
        max_length=max_length,
        segment_budgets=segment_budgets,
    )


def build_cold_start_model(
    model_name: str,
    tokenizer,
    device: torch.device,
    structured_dim: int,
    num_labels: int,
    num_families: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    gradient_checkpointing: bool,
) -> "base.QwenSegmentClassifier":
    """Fresh LoRA + fresh heads, matching train_qwen_segment_v4.py's exact
    cold-start construction. No adapter/heads.pt is loaded from any prior
    checkpoint -- this exists specifically to test hard-negative weighting
    without the 'resuming an already-converged fold0 checkpoint tends to
    overfit further regardless of what's added' confound seen in the V26
    and V28-resumed control runs."""
    backbone = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    backbone.config.use_cache = False
    backbone.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    backbone = get_peft_model(backbone, lora_config)
    for parameter in backbone.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable()

    model = base.QwenSegmentClassifier(
        backbone=backbone,
        hidden_size=backbone.config.hidden_size,
        structured_dim=structured_dim,
        num_labels=num_labels,
        num_families=num_families,
    )
    model.to(device)
    return model


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

    metadata_init = json.loads((args.init_dir / "metadata.json").read_text(encoding="utf-8"))
    postprocess = json.loads((args.init_dir / "postprocess.json").read_text(encoding="utf-8"))
    class_weights_np = np.asarray(postprocess["training_class_weights"], dtype=np.float64)
    teacher_final_logits = base.final_logits_numpy(
        teacher_action_logits,
        teacher_family_logits,
        class_weights_np,
        postprocess,
    ).astype(np.float32)

    sample_weights, distill_weights, hard_summary = build_hard_negative_weights(labels, teacher_final_logits, args)
    print("Hard-negative summary:")
    print(json.dumps({k: v for k, v in hard_summary.items() if k != "per_class"}, ensure_ascii=False, indent=2))
    (args.output_dir / "hard_negative_summary_all_rows.json").write_text(
        json.dumps(hard_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    segment_budgets = base.resolve_segment_budgets(args)
    print("Max length:", args.max_length)
    print("Segment budgets:", segment_budgets)

    if args.smoke:
        selected = balanced_smoke_indices(labels, per_class=args.smoke_per_class)
        samples = [samples[int(i)] for i in selected]
        labels = labels[selected]
        teacher_action_logits = teacher_action_logits[selected]
        teacher_family_logits = teacher_family_logits[selected]
        teacher_final_logits = teacher_final_logits[selected]
        sample_weights = sample_weights[selected]
        distill_weights = distill_weights[selected]
        args.epochs = 1
        args.max_updates = min(args.max_updates or 80, 80)
        print("Smoke samples:", len(samples))

    if args.mode == "eval":
        train_idx, val_idx, groups = make_split(samples, labels, args.eval_fold)
        print(f"Mode=eval fold={args.eval_fold} train={len(train_idx)} validation={len(val_idx)}")
    else:
        train_idx = np.arange(len(labels), dtype=np.int64)
        val_idx = np.asarray([], dtype=np.int64)
        print(f"Mode=full train={len(train_idx)} validation=none")

    model_name = str(metadata_init["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(str(args.init_dir), use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenize datasets...")
    train_dataset = make_hard_dataset(
        samples,
        labels,
        train_idx,
        tokenizer,
        teacher_action_logits,
        teacher_family_logits,
        teacher_final_logits,
        sample_weights,
        distill_weights,
        chunk_size=args.tokenize_chunk_size,
        description="Tokenize V28 hard-negative train",
        max_length=args.max_length,
        segment_budgets=segment_budgets,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=HardNegativeCollator(tokenizer.pad_token_id),
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
            description="Tokenize V28 validation",
            max_length=args.max_length,
            segment_budgets=segment_budgets,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=max(args.validation_batch_size, args.batch_size * 2),
            shuffle=False,
            collate_fn=base.LongSegmentCollator(tokenizer.pad_token_id),
            num_workers=args.num_workers,
            pin_memory=True,
        )

    print("Load V28 student initialized from:", args.init_dir if not args.cold_start else f"COLD START (fresh LoRA, base={model_name})")
    if args.cold_start:
        model = build_cold_start_model(
            model_name,
            tokenizer,
            device,
            structured_dim=STRUCTURED_DIM,
            num_labels=NUM_CLASSES,
            num_families=len(FAMILY_NAMES),
            lora_r=args.cold_start_lora_r,
            lora_alpha=args.cold_start_lora_alpha,
            lora_dropout=args.cold_start_lora_dropout,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    else:
        model = base.load_model_from_init(
            args.init_dir,
            model_name,
            tokenizer,
            device,
            train_lora=not args.freeze_lora,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    optimizer = base.build_optimizer(model, args)

    class_counts = np.bincount(labels[train_idx], minlength=NUM_CLASSES).astype(np.float64)
    ce_weights = np.power(len(train_idx) / (NUM_CLASSES * np.maximum(class_counts, 1.0)), 0.25).astype(np.float32)
    class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    family_index_tensor = torch.tensor(FAMILY_INDEX, dtype=torch.long, device=device)
    ce_weights_tensor = torch.tensor(ce_weights, dtype=torch.float32, device=device)

    total_batches = len(train_loader)
    updates_per_epoch = max(1, (total_batches + args.grad_accum - 1) // args.grad_accum)
    planned_updates = updates_per_epoch * args.epochs
    if args.max_updates and args.max_updates > 0:
        planned_updates = min(planned_updates, args.max_updates)
    warmup_steps = max(1, int(planned_updates * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=planned_updates)
    scaler = torch.amp.GradScaler("cuda")

    initial_eval = None
    best_eval = None
    best_epoch = 0
    if validation_loader is not None:
        initial_eval = base.evaluate(model, validation_loader, device, postprocess, class_weights_tensor, family_index_tensor)
        best_eval = initial_eval
        print(f"Initial init-checkpoint Macro-F1: {initial_eval['macro_f1']:.6f}")

    train_hard_summary = build_hard_negative_weights(labels[train_idx], teacher_final_logits[train_idx], args)[2]
    metadata = {
        "architecture": "qwen_distill_v28_hard_negative",
        "base_model": model_name,
        "initialized_from": "cold_start" if args.cold_start else str(args.init_dir),
        "cold_start": bool(args.cold_start),
        "cold_start_lora": {
            "r": int(args.cold_start_lora_r),
            "alpha": int(args.cold_start_lora_alpha),
            "dropout": float(args.cold_start_lora_dropout),
        } if args.cold_start else None,
        "max_length": int(args.max_length),
        "segment_budgets": segment_budgets,
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
        "hard_negative": {
            "confusion_pairs": args.confusion_pairs,
            "hard_negative_multiplier": float(args.hard_negative_multiplier),
            "confusion_pair_multiplier": float(args.confusion_pair_multiplier),
            "low_margin_threshold": float(args.low_margin_threshold),
            "low_margin_multiplier": float(args.low_margin_multiplier),
            "sample_weight_cap": float(args.sample_weight_cap),
            "sample_weight_min": float(args.sample_weight_min),
            "class_normalized": not bool(args.no_class_normalize_hard_weights),
            "wrong_teacher_distill_scale": float(args.wrong_teacher_distill_scale),
            "pair_teacher_distill_scale": float(args.pair_teacher_distill_scale),
            "all_rows_summary": hard_summary,
            "train_rows_summary": train_hard_summary,
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
        f"Start V28 hard-negative training: epochs={args.epochs}, "
        f"updates_per_epoch={updates_per_epoch}, planned_updates={planned_updates}, grad_accum={args.grad_accum}"
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
            "sample_weight_mean": 0.0,
            "distill_weight_mean": 0.0,
        }
        batch_count = 0
        progress = tqdm(train_loader, desc=f"V28 epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(progress, start=1):
            labels_tensor = batch.pop("labels").to(device, non_blocking=True)
            family_labels = batch.pop("family_labels").to(device, non_blocking=True)
            teacher_action = batch.pop("teacher_action_logits").to(device, non_blocking=True)
            teacher_family = batch.pop("teacher_family_logits").to(device, non_blocking=True)
            teacher_final = batch.pop("teacher_final_logits").to(device, non_blocking=True)
            sample_weight = batch.pop("sample_weight").to(device, non_blocking=True)
            distill_weight = batch.pop("distill_weight").to(device, non_blocking=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                action_logits, family_logits = model(**batch)

            student_final = base.final_logits_torch(action_logits, family_logits, class_weights_tensor, family_index_tensor, postprocess)

            hard_final_loss = weighted_cross_entropy(student_final, labels_tensor, sample_weight, ce_weights_tensor)
            hard_action_loss = weighted_cross_entropy(action_logits, labels_tensor, sample_weight, ce_weights_tensor)
            family_loss = weighted_cross_entropy(family_logits, family_labels, sample_weight, None)
            distill_final_loss = weighted_kl_divergence(student_final, teacher_final, distill_weight, args.distill_temperature)
            distill_action_loss = weighted_kl_divergence(action_logits, teacher_action, distill_weight, args.distill_temperature)
            distill_family_loss = weighted_kl_divergence(family_logits, teacher_family, distill_weight, args.distill_temperature)

            teacher_targets = teacher_final.argmax(dim=1)
            teacher_confidence = F.softmax(teacher_final.float(), dim=1).max(dim=1).values.detach()
            teacher_ce_per = F.cross_entropy(student_final, teacher_targets, reduction="none")
            teacher_ce_loss = weighted_mean(teacher_ce_per, distill_weight * teacher_confidence)

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
            running["sample_weight_mean"] += float(sample_weight.mean().detach().item())
            running["distill_weight_mean"] += float(distill_weight.mean().detach().item())
            batch_count += 1

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=args.max_grad_norm)
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
                    sw=f"{sample_weight.mean().detach().item():.2f}",
                    dw=f"{distill_weight.mean().detach().item():.2f}",
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
            evaluation = base.evaluate(model, validation_loader, device, postprocess, class_weights_tensor, family_index_tensor)
            record["macro_f1"] = float(evaluation["macro_f1"])
            record["improvement"] = float(evaluation["macro_f1"] - initial_eval["macro_f1"])
            print(
                f"V28 epoch {epoch} macro_f1={evaluation['macro_f1']:.6f} "
                f"improvement={evaluation['macro_f1'] - initial_eval['macro_f1']:+.6f} "
                f"loss={record['loss']['total']:.6f}"
            )
            metadata["epoch_records"].append(record)
            if evaluation["macro_f1"] > best_eval["macro_f1"]:
                best_eval = evaluation
                best_epoch = epoch
                metadata["best_epoch"] = int(epoch)
                save_model(model, tokenizer, args.output_dir, metadata, postprocess, evaluation)
        else:
            print(f"V28 epoch {epoch} updates={global_update} loss={record['loss']['total']:.6f}")
            metadata["epoch_records"].append(record)
            metadata["best_epoch"] = int(epoch)
            save_model(model, tokenizer, args.output_dir, metadata, postprocess, None)

        if stop_training:
            print(f"Reached max_updates={args.max_updates}.")
            break

    if validation_loader is not None:
        print()
        print(f"Initial Macro-F1: {initial_eval['macro_f1']:.6f}")
        print(f"Best V28 hard-negative Macro-F1: {best_eval['macro_f1']:.6f}")
        print(f"Improvement: {best_eval['macro_f1'] - initial_eval['macro_f1']:+.6f}")
        print(f"Best epoch: {best_epoch}")
        print()
        print("Class F1 changes:")
        initial_f1 = initial_eval["class_f1"]
        best_f1 = best_eval["class_f1"]
        for index, label in enumerate(ALL_CLASSES):
            print(f"{label:18s} {initial_f1[index]:.6f} -> {best_f1[index]:.6f} ({best_f1[index] - initial_f1[index]:+.6f})")

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
        (args.output_dir / "metadata.json").write_text(json.dumps(final_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        final_metadata = json.loads((args.output_dir / "metadata.json").read_text(encoding="utf-8"))
        final_metadata["final_summary"] = {
            "mode": "full",
            "global_updates": int(global_update),
            "note": "Full training has no local validation score. Submit only after eval-mode validation is promising.",
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(final_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", args.output_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
