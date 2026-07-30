import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from feature_utils_qwen_v4 import ACTION_TO_FAMILY, ALL_CLASSES, FAMILY_NAMES
from train_qwen_segment_v4 import (
    BASE_DIR,
    LABEL2ID,
    EncodedSegmentDataset,
    QwenSegmentClassifier,
    SegmentCollator,
    load_data,
    make_split,
)


MODEL_DIR = Path(BASE_DIR) / "model" / "qwen_segment_v4"
OUTPUT_PATH = MODEL_DIR / "postprocess.json"
LOGITS_PATH = MODEL_DIR / "validation_logits_v4.npz"

BATCH_SIZE = 32


def load_saved_model():
    metadata = json.loads(
        (MODEL_DIR / "metadata.json").read_text(encoding="utf-8")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModel.from_pretrained(
        metadata["base_model"],
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id

    backbone = PeftModel.from_pretrained(
        base_model,
        str(MODEL_DIR / "adapter"),
    )

    model = QwenSegmentClassifier(
        backbone=backbone,
        hidden_size=backbone.config.hidden_size,
        structured_dim=int(metadata["structured_dim"]),
        num_labels=len(ALL_CLASSES),
        num_families=len(FAMILY_NAMES),
    )

    head_state = torch.load(
        MODEL_DIR / "heads.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(head_state, strict=False)
    model.eval().to("cuda")

    return model, tokenizer, metadata


def collect_logits(model, loader):
    action_logits_all: List[np.ndarray] = []
    family_logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Validation logits"):
            labels = batch.pop("labels")
            batch.pop("family_labels")

            batch = {
                key: value.to("cuda", non_blocking=True)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                action_logits, family_logits = model(**batch)

            action_logits_all.append(
                action_logits.float().cpu().numpy()
            )
            family_logits_all.append(
                family_logits.float().cpu().numpy()
            )
            labels_all.append(labels.numpy())

    return (
        np.concatenate(action_logits_all, axis=0),
        np.concatenate(family_logits_all, axis=0),
        np.concatenate(labels_all, axis=0),
    )


def macro_f1(labels, predictions):
    return f1_score(
        labels,
        predictions,
        labels=list(range(len(ALL_CLASSES))),
        average="macro",
        zero_division=0,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다.")

    print("Load data...")
    samples, targets = load_data()
    train_indices, validation_indices, labels = make_split(
        samples,
        targets,
    )

    validation_samples = [
        samples[index] for index in validation_indices
    ]
    validation_labels = labels[validation_indices]
    train_labels = labels[train_indices]

    model, tokenizer, metadata = load_saved_model()

    dataset = EncodedSegmentDataset(
        validation_samples,
        validation_labels,
        tokenizer,
        chunk_size=2048,
        description="Tokenize validation",
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=SegmentCollator(tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=True,
    )

    action_logits, family_logits, y_true = collect_logits(
        model,
        loader,
    )

    np.savez_compressed(
        LOGITS_PATH,
        action_logits=action_logits.astype(np.float32),
        family_logits=family_logits.astype(np.float32),
        labels=y_true.astype(np.int64),
        validation_indices=validation_indices.astype(np.int64),
    )

    class_counts = np.bincount(
        train_labels,
        minlength=len(ALL_CLASSES),
    ).astype(np.float64)

    training_class_weights = np.power(
        len(train_labels)
        / (
            len(ALL_CLASSES)
            * np.maximum(class_counts, 1.0)
        ),
        0.25,
    )
    log_training_weights = np.log(
        np.maximum(training_class_weights, 1e-12)
    )

    family_index = np.asarray(
        ACTION_TO_FAMILY,
        dtype=np.int64,
    )

    raw_predictions = action_logits.argmax(axis=1)
    raw_score = macro_f1(y_true, raw_predictions)

    print(f"Raw action-head Macro-F1: {raw_score:.6f}")

    best = {
        "macro_f1": raw_score,
        "family_weight": 0.0,
        "prior_beta": 0.0,
        "action_temperature": 1.0,
    }
    best_predictions = raw_predictions

    family_weights = np.round(
        np.arange(0.0, 1.5001, 0.05),
        2,
    )
    prior_betas = np.round(
    np.arange(-1.00, 0.5001, 0.05),
    2,
    )

    action_temperatures = [
     0.80,
     1.00,
      1.20,
      1.40,
     1.60,
      1.80,
      2.00,
    ]

    for action_temperature in action_temperatures:
        scaled_action = action_logits / action_temperature

        for family_weight in family_weights:
            family_adjustment = (
                family_weight
                * family_logits[:, family_index]
            )

            for prior_beta in prior_betas:
                final_logits = (
                    scaled_action
                    + family_adjustment
                    - prior_beta
                    * log_training_weights[None, :]
                )
                predictions = final_logits.argmax(axis=1)
                score = macro_f1(y_true, predictions)

                if score > best["macro_f1"]:
                    best = {
                        "macro_f1": float(score),
                        "family_weight": float(family_weight),
                        "prior_beta": float(prior_beta),
                        "action_temperature": float(
                            action_temperature
                        ),
                    }
                    best_predictions = predictions

    print()
    print("Best postprocess:")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    print()
    print(
        classification_report(
            y_true,
            best_predictions,
            labels=list(range(len(ALL_CLASSES))),
            target_names=ALL_CLASSES,
            digits=4,
            zero_division=0,
        )
    )

    result = {
        **best,
        "classes": ALL_CLASSES,
        "action_to_family": ACTION_TO_FAMILY,
        "training_class_weights": (
            training_class_weights.tolist()
        ),
        "source_validation_macro_f1": float(
            metadata.get("validation_macro_f1", -1.0)
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Saved:", OUTPUT_PATH)
    print("Saved:", LOGITS_PATH)


if __name__ == "__main__":
    main()
