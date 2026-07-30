from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from feature_utils_qwen_v4 import ACTION_TO_FAMILY, ALL_CLASSES

NUM_CLASSES = len(ALL_CLASSES)
FAMILY_INDEX = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)


def macro_f1(y_true, y_pred):
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0,
        )
    )


def final_logits(action_logits, family_logits, class_weights, temperature, family_weight, prior_beta):
    return (
        action_logits / temperature
        + family_weight * family_logits[:, FAMILY_INDEX]
        - prior_beta * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )


def parse_grid(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--base-postprocess", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperatures", default="1.0,1.2,1.4,1.6,1.8")
    parser.add_argument("--family-weights", default="0.75,1.0,1.25,1.5,1.75")
    parser.add_argument("--prior-betas", default="-1.10,-0.95,-0.85,-0.70,-0.50")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.logits)
    action_logits = npz["action_logits"].astype(np.float64)
    family_logits = npz["family_logits"].astype(np.float64)
    labels = npz["labels"].astype(np.int64)

    base = json.loads(args.base_postprocess.read_text(encoding="utf-8"))
    class_weights = np.asarray(base["training_class_weights"], dtype=np.float64)

    base_cfg = (
        float(base["action_temperature"]),
        float(base["family_weight"]),
        float(base["prior_beta"]),
    )

    configs = list(
        product(
            parse_grid(args.temperatures),
            parse_grid(args.family_weights),
            parse_grid(args.prior_betas),
        )
    )
    if base_cfg not in configs:
        configs.append(base_cfg)

    pred_cache = {}
    full_scores = {}
    for cfg in configs:
        pred = final_logits(
            action_logits,
            family_logits,
            class_weights,
            cfg[0],
            cfg[1],
            cfg[2],
        ).argmax(axis=1)
        pred_cache[cfg] = pred
        full_scores[cfg] = macro_f1(labels, pred)

    base_pred = pred_cache[base_cfg]
    base_full = full_scores[base_cfg]

    splitter = RepeatedStratifiedKFold(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.seed,
    )

    chosen = []
    gains = []
    split_rows = []

    for split_id, (tune_idx, eval_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        best_cfg = max(
            configs,
            key=lambda cfg: macro_f1(labels[tune_idx], pred_cache[cfg][tune_idx]),
        )
        chosen.append(best_cfg)

        base_eval = macro_f1(labels[eval_idx], base_pred[eval_idx])
        candidate_eval = macro_f1(labels[eval_idx], pred_cache[best_cfg][eval_idx])
        gain = candidate_eval - base_eval
        gains.append(gain)

        split_rows.append(
            {
                "split": split_id,
                "temperature": best_cfg[0],
                "family_weight": best_cfg[1],
                "prior_beta": best_cfg[2],
                "base_eval_f1": base_eval,
                "candidate_eval_f1": candidate_eval,
                "gain": gain,
            }
        )

    counts = Counter(chosen)
    recommended_cfg, recommended_count = counts.most_common(1)[0]
    gains_arr = np.asarray(gains, dtype=np.float64)

    nested = {
        "gain_mean": float(gains_arr.mean()),
        "gain_min": float(gains_arr.min()),
        "gain_max": float(gains_arr.max()),
        "positive": int((gains_arr > 0).sum()),
        "total": int(len(gains_arr)),
    }

    full_best_cfg = max(configs, key=full_scores.get)
    full_best_score = full_scores[full_best_cfg]

    verdict = "DISCARD"
    if nested["positive"] >= 20 and nested["gain_mean"] >= 0.0005:
        verdict = "PROMISING"
    elif nested["positive"] >= 15 and nested["gain_mean"] > 0:
        verdict = "MARGINAL"

    recommended_post = dict(base)
    recommended_post["action_temperature"] = float(recommended_cfg[0])
    recommended_post["family_weight"] = float(recommended_cfg[1])
    recommended_post["prior_beta"] = float(recommended_cfg[2])

    report = {
        "base_config": {
            "action_temperature": base_cfg[0],
            "family_weight": base_cfg[1],
            "prior_beta": base_cfg[2],
            "full_f1": base_full,
        },
        "full_data_best_reference_only": {
            "action_temperature": full_best_cfg[0],
            "family_weight": full_best_cfg[1],
            "prior_beta": full_best_cfg[2],
            "full_f1": full_best_score,
            "gain": full_best_score - base_full,
        },
        "nested_selection": nested,
        "recommended_mode_config": {
            "action_temperature": recommended_cfg[0],
            "family_weight": recommended_cfg[1],
            "prior_beta": recommended_cfg[2],
            "selected_count": recommended_count,
            "full_f1": full_scores[recommended_cfg],
            "full_gain": full_scores[recommended_cfg] - base_full,
        },
        "verdict": verdict,
        "selection_counts": [
            {
                "action_temperature": cfg[0],
                "family_weight": cfg[1],
                "prior_beta": cfg[2],
                "count": count,
            }
            for cfg, count in counts.most_common()
        ],
        "splits": split_rows,
    }

    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "recommended_postprocess.json").write_text(
        json.dumps(recommended_post, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Base full Macro-F1: {base_full:.6f}")
    print(
        f"Full-data best (reference only): T={full_best_cfg[0]:.2f}, "
        f"F={full_best_cfg[1]:.2f}, P={full_best_cfg[2]:.2f}, "
        f"F1={full_best_score:.6f}, gain={full_best_score - base_full:+.6f}"
    )
    print(
        f"Nested selection: gain_mean={nested['gain_mean']:+.6f}, "
        f"gain_min={nested['gain_min']:+.6f}, "
        f"positive={nested['positive']}/{nested['total']}"
    )
    print(
        f"Recommended mode config: T={recommended_cfg[0]:.2f}, "
        f"F={recommended_cfg[1]:.2f}, P={recommended_cfg[2]:.2f}, "
        f"selected={recommended_count}/{len(chosen)}, "
        f"full_gain={full_scores[recommended_cfg] - base_full:+.6f}"
    )
    print("Verdict:", verdict)
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
