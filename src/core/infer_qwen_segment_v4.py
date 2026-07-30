import csv
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from feature_utils_qwen_v4 import (
    ACTION_TO_FAMILY,
    ALL_CLASSES,
    FAMILY_NAMES,
    STRUCTURED_DIM,
)
from train_qwen_segment_v4 import (
    EncodedSegmentDataset,
    QwenSegmentClassifier,
    SegmentCollator,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = (
    BASE_DIR / "model" / "qwen_segment_v4"
)
OUTPUT_PATH = (
    BASE_DIR
    / "output"
    / "submission_qwen_v4.csv"
)

BATCH_SIZE = 64


def load_jsonl(path: Path) -> List[dict]:
    samples: List[dict] = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                samples.append(json.loads(line))

    return samples


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU가 필요합니다."
        )

    metadata = json.loads(
        (MODEL_DIR / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    base_model = AutoModel.from_pretrained(
        metadata["base_model"],
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = False
    base_model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    backbone = PeftModel.from_pretrained(
        base_model,
        str(MODEL_DIR / "adapter"),
    )

    model = QwenSegmentClassifier(
        backbone=backbone,
        hidden_size=backbone.config.hidden_size,
        structured_dim=int(
            metadata["structured_dim"]
        ),
        num_labels=len(ALL_CLASSES),
        num_families=len(FAMILY_NAMES),
    )

    state = torch.load(
        MODEL_DIR / "heads.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(
        state,
        strict=False,
    )

    model.eval().to("cuda")

    samples = load_jsonl(
        DATA_DIR / "test.jsonl"
    )
    ids = [
        str(sample.get("id", ""))
        for sample in samples
    ]

    dummy_labels = np.zeros(
        len(samples),
        dtype=np.int64,
    )

    dataset = EncodedSegmentDataset(
        samples,
        dummy_labels,
        tokenizer,
        chunk_size=2048,
        description="Tokenize test",
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=SegmentCollator(
            tokenizer.pad_token_id
        ),
        num_workers=0,
        pin_memory=True,
    )

    family_index = torch.tensor(
        ACTION_TO_FAMILY,
        dtype=torch.long,
        device="cuda",
    )
    family_logit_weight = float(
        metadata["family_logit_weight"]
    )

    predictions: List[int] = []

    started = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels")
            batch.pop("family_labels")

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
                (
                    action_logits,
                    family_logits,
                ) = model(**batch)

                final_logits = (
                    action_logits
                    + family_logit_weight
                    * family_logits[
                        :,
                        family_index,
                    ]
                )

            predictions.extend(
                final_logits.argmax(
                    dim=-1
                ).cpu().tolist()
            )

    torch.cuda.synchronize()
    elapsed = (
        time.perf_counter()
        - started
    )

    prediction_map = {
        sample_id: ALL_CLASSES[prediction]
        for sample_id, prediction in zip(
            ids,
            predictions,
        )
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

    for row in rows:
        row["action"] = prediction_map[
            row["id"]
        ]

    OUTPUT_PATH.parent.mkdir(
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

    print(
        f"Saved: {OUTPUT_PATH} "
        f"rows={len(rows)}"
    )
    print(
        f"Inference seconds: "
        f"{elapsed:.3f}"
    )


if __name__ == "__main__":
    main()
