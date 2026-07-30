from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from feature_utils_qwen_v4 import ACTION_TO_FAMILY, ALL_CLASSES

NUM_CLASSES = len(ALL_CLASSES)
FAMILY_INDEX = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0,
        )
    )


def parse_alphas(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one alpha is required.")
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise argparse.ArgumentTypeError("Alphas must be between 0 and 1.")
    return values


def compute_final(
    action_logits: np.ndarray,
    family_logits: np.ndarray,
    postprocess: dict,
) -> np.ndarray:
    class_weights = np.asarray(
        postprocess["training_class_weights"],
        dtype=np.float64,
    )
    return (
        action_logits.astype(np.float64)
        / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"])
        * family_logits.astype(np.float64)[:, FAMILY_INDEX]
        - float(postprocess["prior_beta"])
        * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Screen global blends between the original current96 Qwen OOF "
            "teacher and the cur160 OOF teacher, then optionally save the "
            "best blended teacher NPZ for V12 distillation."
        )
    )
    parser.add_argument("--original-oof", type=Path, required=True)
    parser.add_argument("--cur160-oof", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alphas",
        type=parse_alphas,
        default=parse_alphas("0,0.25,0.5,0.65,0.75,0.85,0.9,1.0"),
        help="Cur160 weight. 0=original only, 1=cur160 only.",
    )
    parser.add_argument(
        "--min-folds-nondecreasing",
        type=int,
        default=4,
        help="Minimum folds that must not be worse than cur160-only.",
    )
    parser.add_argument(
        "--min-full-gain",
        type=float,
        default=0.0002,
        help="Minimum full OOF gain over cur160-only to mark promising.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    original = np.load(args.original_oof)
    cur160 = np.load(args.cur160_oof)
    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))

    required = {"action_logits", "family_logits", "labels"}
    for name, npz in (("original", original), ("cur160", cur160)):
        missing = required - set(npz.files)
        if missing:
            raise RuntimeError(f"{name} OOF is missing keys: {sorted(missing)}")

    labels_original = original["labels"].astype(np.int64)
    labels_cur160 = cur160["labels"].astype(np.int64)
    if not np.array_equal(labels_original, labels_cur160):
        raise RuntimeError("OOF labels are not aligned.")

    labels = labels_cur160
    fold_ids = (
        cur160["fold_ids"].astype(np.int64)
        if "fold_ids" in cur160.files
        else None
    )

    original_action = original["action_logits"].astype(np.float64)
    original_family = original["family_logits"].astype(np.float64)
    cur160_action = cur160["action_logits"].astype(np.float64)
    cur160_family = cur160["family_logits"].astype(np.float64)

    expected_action_shape = (len(labels), NUM_CLASSES)
    if original_action.shape != expected_action_shape:
        raise RuntimeError(
            f"Original action shape {original_action.shape} != {expected_action_shape}"
        )
    if cur160_action.shape != expected_action_shape:
        raise RuntimeError(
            f"Cur160 action shape {cur160_action.shape} != {expected_action_shape}"
        )
    if original_family.shape != cur160_family.shape:
        raise RuntimeError(
            f"Family-logit shapes differ: {original_family.shape} vs {cur160_family.shape}"
        )

    rows = []
    cache = {}

    for alpha in args.alphas:
        action = (
            (1.0 - alpha) * original_action
            + alpha * cur160_action
        )
        family = (
            (1.0 - alpha) * original_family
            + alpha * cur160_family
        )
        final = compute_final(action, family, postprocess)
        pred = final.argmax(axis=1).astype(np.int64)
        full_f1 = macro_f1(labels, pred)

        fold_scores = {}
        if fold_ids is not None:
            for fold in sorted(np.unique(fold_ids).tolist()):
                idx = np.flatnonzero(fold_ids == fold)
                fold_scores[str(int(fold))] = macro_f1(labels[idx], pred[idx])

        rows.append(
            {
                "alpha_cur160": float(alpha),
                "full_macro_f1": float(full_f1),
                "fold_macro_f1": fold_scores,
            }
        )
        cache[float(alpha)] = (action, family, final)

    cur160_row = min(
        rows,
        key=lambda row: abs(row["alpha_cur160"] - 1.0),
    )
    cur160_score = float(cur160_row["full_macro_f1"])
    cur160_fold_scores = cur160_row["fold_macro_f1"]

    for row in rows:
        row["full_gain_vs_cur160"] = (
            float(row["full_macro_f1"]) - cur160_score
        )
        nondecreasing = None
        if fold_ids is not None:
            nondecreasing = sum(
                row["fold_macro_f1"][fold]
                >= cur160_fold_scores[fold] - 1e-12
                for fold in cur160_fold_scores
            )
        row["folds_nondecreasing_vs_cur160"] = nondecreasing

    best = max(rows, key=lambda row: row["full_macro_f1"])
    best_alpha = float(best["alpha_cur160"])
    best_action, best_family, best_final = cache[best_alpha]

    promising = (
        best["full_gain_vs_cur160"] >= args.min_full_gain
        and (
            fold_ids is None
            or best["folds_nondecreasing_vs_cur160"]
            >= args.min_folds_nondecreasing
        )
        and best_alpha < 1.0
    )

    output_npz = args.output_dir / "oof_logits_blend_best.npz"
    save_payload = {
        "action_logits": best_action.astype(np.float32),
        "family_logits": best_family.astype(np.float32),
        "final_logits": best_final.astype(np.float32),
        "labels": labels.astype(np.int64),
        "alpha_cur160": np.asarray(best_alpha, dtype=np.float32),
    }
    if fold_ids is not None:
        save_payload["fold_ids"] = fold_ids.astype(np.int8)
    np.savez_compressed(output_npz, **save_payload)

    report = {
        "original_oof": str(args.original_oof),
        "cur160_oof": str(args.cur160_oof),
        "postprocess": str(args.postprocess),
        "cur160_only_full_macro_f1": cur160_score,
        "best": best,
        "promising": bool(promising),
        "rows": rows,
        "output_npz": str(output_npz),
    }
    (args.output_dir / "teacher_blend_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Cur160-only OOF Macro-F1: {cur160_score:.6f}")
    print()
    for row in sorted(rows, key=lambda x: x["alpha_cur160"]):
        fold_text = ""
        if row["folds_nondecreasing_vs_cur160"] is not None:
            fold_text = (
                f", folds>=cur160="
                f"{row['folds_nondecreasing_vs_cur160']}/5"
            )
        print(
            f"alpha_cur160={row['alpha_cur160']:.2f} "
            f"F1={row['full_macro_f1']:.6f} "
            f"gain={row['full_gain_vs_cur160']:+.6f}"
            f"{fold_text}"
        )

    print()
    print(
        f"Best alpha_cur160={best_alpha:.2f}, "
        f"F1={best['full_macro_f1']:.6f}, "
        f"gain={best['full_gain_vs_cur160']:+.6f}"
    )
    print("Verdict:", "PROMISING" if promising else "DISCARD / WEAK")
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
