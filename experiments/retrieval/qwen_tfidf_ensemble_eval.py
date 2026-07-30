import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from feature_utils_qwen_v3 import ALL_CLASSES, build_prompt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
QWEN_MODEL_DIR = BASE_DIR / "model" / "qwen_pool_v3"
TFIDF_MODEL_PATH = BASE_DIR / "model" / "tfidf_sgd_ensemble.pkl"
CONFIG_PATH = BASE_DIR / "model" / "ensemble_config.json"
OOF_PATH = BASE_DIR / "model" / "ensemble_validation_outputs.npz"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-batch-size", type=int, default=64)
    parser.add_argument("--word-features", type=int, default=100_000)
    parser.add_argument("--char-features", type=int, default=180_000)
    parser.add_argument("--alpha", type=float, default=3e-6)
    return parser.parse_args()


def load_data() -> Tuple[List[dict], np.ndarray]:
    samples: List[dict] = []
    with (DATA_DIR / "train.jsonl").open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    with (DATA_DIR / "train_labels.csv").open(encoding="utf-8", newline="") as file:
        label_map = {
            row["id"]: row["action"]
            for row in csv.DictReader(file)
        }

    label_to_id = {
        label: index
        for index, label in enumerate(ALL_CLASSES)
    }
    labels = np.array(
        [label_to_id[label_map[sample["id"]]] for sample in samples],
        dtype=np.int64,
    )
    return samples, labels


def make_split(samples: List[dict], labels: np.ndarray):
    groups = np.array([
        str(sample["id"]).rsplit("-step_", 1)[0]
        for sample in samples
    ])
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )
    train_indices, validation_indices = next(
        splitter.split(np.zeros(len(labels)), labels, groups)
    )
    overlap = set(groups[train_indices]) & set(groups[validation_indices])
    if overlap:
        raise RuntimeError(f"학습·검증 세션이 {len(overlap)}개 겹칩니다.")
    return train_indices, validation_indices


def make_tfidf_model(word_features: int, char_features: int, alpha: float) -> Pipeline:
    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_features=word_features,
                sublinear_tf=True,
                lowercase=True,
                dtype=np.float32,
                token_pattern=r"(?u)\b\w+\b",
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_features=char_features,
                sublinear_tf=True,
                lowercase=True,
                dtype=np.float32,
            ),
        ),
    ])
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=40,
        tol=1e-4,
        class_weight="balanced",
        average=True,
        random_state=SEED,
        n_jobs=-1,
    )
    return Pipeline([
        ("features", features),
        ("classifier", classifier),
    ])


class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, max_length):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.tokenizer(
            self.prompts[index],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )


class PromptCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        return self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )


class QwenMeanPoolClassifier(nn.Module):
    def __init__(self, backbone, hidden_size, num_labels):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.0)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.norm(pooled.float())
        return self.classifier(self.dropout(pooled))


def load_qwen_model():
    if not QWEN_MODEL_DIR.exists():
        raise FileNotFoundError(f"Qwen 모델 폴더가 없습니다: {QWEN_MODEL_DIR}")

    metadata = json.loads(
        (QWEN_MODEL_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(QWEN_MODEL_DIR),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    base_model = AutoModel.from_pretrained(
        metadata["base_model"],
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id

    backbone = PeftModel.from_pretrained(
        base_model,
        str(QWEN_MODEL_DIR / "adapter"),
    )
    model = QwenMeanPoolClassifier(
        backbone=backbone,
        hidden_size=backbone.config.hidden_size,
        num_labels=len(ALL_CLASSES),
    )
    head = torch.load(
        QWEN_MODEL_DIR / "head.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.norm.load_state_dict(head["norm"])
    model.classifier.load_state_dict(head["classifier"])
    model.eval().to("cuda")
    return model, tokenizer, int(metadata["max_length"])


def predict_qwen_logits(prompts: List[str], batch_size: int) -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen 검증에는 CUDA GPU가 필요합니다.")

    model, tokenizer, max_length = load_qwen_model()
    dataset = PromptDataset(prompts, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PromptCollator(tokenizer),
        num_workers=0,
        pin_memory=True,
    )
    logits_list = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Qwen validation"):
            batch = {
                key: value.to("cuda", non_blocking=True)
                for key, value in batch.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**batch)
            logits_list.append(logits.float().cpu().numpy())
    return np.concatenate(logits_list, axis=0)


def aligned_tfidf_probabilities(model: Pipeline, texts: List[str]) -> np.ndarray:
    probabilities = model.predict_proba(texts)
    classifier_classes = model.named_steps["classifier"].classes_
    aligned = np.zeros((len(texts), len(ALL_CLASSES)), dtype=np.float32)
    for source_index, class_id in enumerate(classifier_classes):
        aligned[:, int(class_id)] = probabilities[:, source_index]
    return aligned


def search_blend(labels, qwen_logits, tfidf_probabilities):
    temperatures = [0.50, 0.65, 0.80, 1.00, 1.20, 1.50, 2.00]
    qwen_weights = np.arange(0.0, 1.0001, 0.05)

    best = {"macro_f1": -1.0, "temperature": 1.0, "qwen_weight": 1.0}
    qwen_only_best = {"macro_f1": -1.0, "temperature": 1.0}

    for temperature in temperatures:
        qwen_probabilities = softmax(qwen_logits / temperature, axis=1)
        qwen_predictions = qwen_probabilities.argmax(axis=1)
        qwen_f1 = f1_score(
            labels,
            qwen_predictions,
            labels=list(range(len(ALL_CLASSES))),
            average="macro",
            zero_division=0,
        )
        if qwen_f1 > qwen_only_best["macro_f1"]:
            qwen_only_best = {
                "macro_f1": float(qwen_f1),
                "temperature": float(temperature),
            }

        for qwen_weight in qwen_weights:
            blend = (
                qwen_weight * qwen_probabilities
                + (1.0 - qwen_weight) * tfidf_probabilities
            )
            predictions = blend.argmax(axis=1)
            score = f1_score(
                labels,
                predictions,
                labels=list(range(len(ALL_CLASSES))),
                average="macro",
                zero_division=0,
            )
            if score > best["macro_f1"]:
                best = {
                    "macro_f1": float(score),
                    "temperature": float(temperature),
                    "qwen_weight": float(qwen_weight),
                }
    return best, qwen_only_best


def main():
    args = parse_args()
    print("Load data...")
    samples, labels = load_data()
    train_indices, validation_indices = make_split(samples, labels)

    texts = [build_prompt(sample) for sample in samples]
    train_texts = [texts[index] for index in train_indices]
    validation_texts = [texts[index] for index in validation_indices]
    train_labels = labels[train_indices]
    validation_labels = labels[validation_indices]
    print(f"train={len(train_texts)} val={len(validation_texts)}")

    print("Train TF-IDF model...")
    tfidf_model = make_tfidf_model(
        args.word_features,
        args.char_features,
        args.alpha,
    )
    tfidf_model.fit(train_texts, train_labels)
    TFIDF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf_model, TFIDF_MODEL_PATH, compress=3)
    print("Saved TF-IDF:", TFIDF_MODEL_PATH)

    tfidf_probabilities = aligned_tfidf_probabilities(
        tfidf_model,
        validation_texts,
    )
    tfidf_predictions = tfidf_probabilities.argmax(axis=1)
    tfidf_f1 = f1_score(
        validation_labels,
        tfidf_predictions,
        labels=list(range(len(ALL_CLASSES))),
        average="macro",
        zero_division=0,
    )
    print(f"TF-IDF Macro-F1: {tfidf_f1:.6f}")

    print("Run Qwen on the same validation fold...")
    qwen_logits = predict_qwen_logits(
        validation_texts,
        batch_size=args.qwen_batch_size,
    )

    best, qwen_only_best = search_blend(
        validation_labels,
        qwen_logits,
        tfidf_probabilities,
    )
    print(
        f"Best Qwen-only Macro-F1: {qwen_only_best['macro_f1']:.6f} "
        f"temperature={qwen_only_best['temperature']}"
    )
    print(
        f"Best ensemble Macro-F1: {best['macro_f1']:.6f} "
        f"qwen_weight={best['qwen_weight']:.2f} "
        f"tfidf_weight={1.0 - best['qwen_weight']:.2f} "
        f"temperature={best['temperature']}"
    )

    qwen_probabilities = softmax(
        qwen_logits / best["temperature"],
        axis=1,
    )
    final_probabilities = (
        best["qwen_weight"] * qwen_probabilities
        + (1.0 - best["qwen_weight"]) * tfidf_probabilities
    )
    final_predictions = final_probabilities.argmax(axis=1)
    print(
        classification_report(
            validation_labels,
            final_predictions,
            labels=list(range(len(ALL_CLASSES))),
            target_names=ALL_CLASSES,
            digits=4,
            zero_division=0,
        )
    )

    config = {
        "classes": ALL_CLASSES,
        "qwen_weight": best["qwen_weight"],
        "tfidf_weight": 1.0 - best["qwen_weight"],
        "qwen_temperature": best["temperature"],
        "validation_macro_f1": best["macro_f1"],
        "qwen_only_macro_f1": qwen_only_best["macro_f1"],
        "tfidf_only_macro_f1": float(tfidf_f1),
        "split_seed": SEED,
        "split_n_splits": 5,
        "split_fold": 0,
        "tfidf_word_features": args.word_features,
        "tfidf_char_features": args.char_features,
        "tfidf_alpha": args.alpha,
    }
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        OOF_PATH,
        validation_indices=validation_indices,
        validation_labels=validation_labels,
        qwen_logits=qwen_logits.astype(np.float32),
        tfidf_probabilities=tfidf_probabilities.astype(np.float32),
        ensemble_probabilities=final_probabilities.astype(np.float32),
    )
    print("Saved config:", CONFIG_PATH)
    print("Saved validation outputs:", OOF_PATH)


if __name__ == "__main__":
    main()
