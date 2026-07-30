#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V30 gated blend holdout check. No training.

Example:
python .\probe_v30_gated_holdout.py `
  --data .\data\train.jsonl `
  --labels-csv .\data\train_labels.csv `
  --feature-utils .\feature_utils_qwen_v4_enflags.py `
  --qwen-logits .\model\qwen_distill_v12_eval\validation_logits_v12.npz `
  --postprocess .\model\qwen_distill_v12_eval\postprocess.json `
  --tree-proba .\model\v30_enflags_tree_eval\tree_prob_val.npy `
  --val-indices .\model\v30_enflags_tree_eval\val_indices.npy `
  --output-dir .\model\v30_gated_holdout_probe
"""
import argparse, csv, importlib.util, json, re, sys
from pathlib import Path
from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_RE = re.compile(r"[A-Za-z]")

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_feature_utils(path: Path):
    spec = importlib.util.spec_from_file_location("feature_utils_v30_probe", str(path.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load feature utils: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def read_labels(labels_csv: Path, samples: List[dict], classes: List[str]) -> np.ndarray:
    label_to_id = {c: i for i, c in enumerate(classes)}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        label_map = {str(row["id"]): str(row["action"]) for row in csv.DictReader(f)}
    return np.asarray([label_to_id[label_map[str(s["id"])]] for s in samples], dtype=np.int64)

def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)

def load_qwen_proba(logits_path: Path, postprocess_path: Path, fu) -> np.ndarray:
    z = np.load(logits_path)
    final = z["action_logits"].astype(np.float64)
    post = json.loads(postprocess_path.read_text(encoding="utf-8"))
    final = final / float(post.get("action_temperature", 1.0))
    family_weight = float(post.get("family_weight", 0.0))
    if "family_logits" in z and family_weight != 0.0:
        family_index = np.asarray(fu.ACTION_TO_FAMILY, dtype=np.int64)
        final += family_weight * z["family_logits"].astype(np.float64)[:, family_index]
    prior_beta = float(post.get("prior_beta", 0.0))
    class_weights = post.get("training_class_weights")
    if class_weights is not None and prior_beta != 0.0:
        cw = np.asarray(class_weights, dtype=np.float64)
        final -= prior_beta * np.log(np.maximum(cw, 1e-12))[None, :]
    return stable_softmax(final)

def align_qwen(qwen: np.ndarray, full_n: int, val_idx: np.ndarray) -> np.ndarray:
    if qwen.shape[0] == full_n:
        return qwen[val_idx]
    if qwen.shape[0] == len(val_idx):
        return qwen
    raise RuntimeError(f"Cannot align qwen rows={qwen.shape[0]} full={full_n} val={len(val_idx)}")

def language_bucket(text: str) -> str:
    hangul = len(_HANGUL_RE.findall(text or ""))
    latin = len(_LATIN_RE.findall(text or ""))
    if hangul >= 3 and latin >= 10:
        return "mixed"
    if hangul >= 3:
        return "ko"
    if latin >= 10:
        return "en"
    if hangul > 0 and latin > 0:
        return "mixed"
    if hangul > 0:
        return "ko"
    if latin > 0:
        return "en"
    return "other"

def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    return float(f1_score(y, pred, labels=np.arange(n_classes), average="macro", zero_division=0))

def global_pred(base: np.ndarray, tree: np.ndarray, w: float) -> np.ndarray:
    scores = (1.0 - w) * np.log(np.maximum(base, 1e-12)) + w * np.log(np.maximum(tree, 1e-12))
    return scores.argmax(axis=1)

def gated_pred(base: np.ndarray, tree: np.ndarray, langs: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
    log_base = np.log(np.maximum(base, 1e-12))
    log_tree = np.log(np.maximum(tree, 1e-12))
    scores = log_base.copy()
    for lang, w in weights.items():
        mask = langs == lang
        if mask.any() and w > 0.0:
            scores[mask] = (1.0 - w) * log_base[mask] + w * log_tree[mask]
    return scores.argmax(axis=1)

def parse_weights(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]

def summarize_by_lang(y, pred, langs, n_classes):
    rows = []
    for lang in ["mixed", "ko", "en", "other"]:
        mask = langs == lang
        if not mask.any():
            continue
        rows.append({
            "lang": lang,
            "n": int(mask.sum()),
            "macro_f1": macro_f1(y[mask], pred[mask], n_classes),
            "accuracy": float(accuracy_score(y[mask], pred[mask])),
            "error_rate": float((y[mask] != pred[mask]).mean()),
            "wrong": int((y[mask] != pred[mask]).sum()),
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--feature-utils", type=Path, required=True)
    ap.add_argument("--qwen-logits", type=Path, required=True)
    ap.add_argument("--postprocess", type=Path, required=True)
    ap.add_argument("--tree-proba", type=Path, required=True)
    ap.add_argument("--val-indices", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--global-weight", type=float, default=0.05)
    ap.add_argument("--en-weights", type=parse_weights, default=parse_weights("0,0.05,0.10,0.15,0.20,0.25,0.30,0.35"))
    ap.add_argument("--mixed-weights", type=parse_weights, default=parse_weights("0,0.05,0.10,0.15,0.20"))
    ap.add_argument("--ko-weights", type=parse_weights, default=parse_weights("0,0.05,0.10,0.15,0.20"))
    ap.add_argument("--other-weight", type=float, default=0.0)
    ap.add_argument("--holdout-seeds", type=int, default=30)
    ap.add_argument("--holdout-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fu = load_feature_utils(args.feature_utils)
    classes = list(fu.ALL_CLASSES)
    n_classes = len(classes)
    samples = load_jsonl(args.data)
    labels_all = read_labels(args.labels_csv, samples, classes)
    val_idx = np.load(args.val_indices).astype(np.int64)
    y = labels_all[val_idx]

    qwen_all = load_qwen_proba(args.qwen_logits, args.postprocess, fu)
    qwen = align_qwen(qwen_all, len(samples), val_idx)
    tree = np.load(args.tree_proba).astype(np.float64)
    if tree.shape[0] != len(val_idx):
        raise RuntimeError(f"tree rows={tree.shape[0]} does not match val rows={len(val_idx)}")

    val_samples = [samples[i] for i in val_idx]
    langs = np.asarray([language_bucket(str(s.get("current_prompt", ""))) for s in val_samples])

    base_pred = qwen.argmax(axis=1)
    global_p = global_pred(qwen, tree, args.global_weight)
    base_f1 = macro_f1(y, base_pred, n_classes)
    global_f1 = macro_f1(y, global_p, n_classes)

    print(f"Base Qwen Macro-F1:   {base_f1:.6f}")
    print(f"Global w={args.global_weight:.3f}: {global_f1:.6f} gain={global_f1 - base_f1:+.6f}")

    candidates = []
    for w_en in args.en_weights:
        for w_mixed in args.mixed_weights:
            for w_ko in args.ko_weights:
                weights = {"en": float(w_en), "mixed": float(w_mixed), "ko": float(w_ko), "other": float(args.other_weight)}
                pred = gated_pred(qwen, tree, langs, weights)
                f1 = macro_f1(y, pred, n_classes)
                candidates.append({
                    "weights": weights,
                    "full_macro_f1": float(f1),
                    "full_gain_vs_qwen": float(f1 - base_f1),
                    "full_gain_vs_global": float(f1 - global_f1),
                })

    candidates.sort(key=lambda r: r["full_macro_f1"], reverse=True)
    print("\nTop gated candidates on full fold-val:")
    for row in candidates[:15]:
        print(f"  weights={row['weights']} f1={row['full_macro_f1']:.6f} gain_vs_qwen={row['full_gain_vs_qwen']:+.6f} gain_vs_global={row['full_gain_vs_global']:+.6f}")

    rng_master = np.random.default_rng(args.seed)
    records = []
    for rank, cand in enumerate(candidates[:30], start=1):
        gains_vs_qwen, gains_vs_global = [], []
        positive_vs_qwen, positive_vs_global = 0, 0
        weights = cand["weights"]
        pred_full = gated_pred(qwen, tree, langs, weights)
        for _ in range(args.holdout_seeds):
            rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
            idx = rng.choice(len(y), size=max(1, int(len(y) * args.holdout_frac)), replace=False)
            q_f1 = macro_f1(y[idx], base_pred[idx], n_classes)
            g_f1 = macro_f1(y[idx], global_p[idx], n_classes)
            c_f1 = macro_f1(y[idx], pred_full[idx], n_classes)
            gains_vs_qwen.append(c_f1 - q_f1)
            gains_vs_global.append(c_f1 - g_f1)
            positive_vs_qwen += int(c_f1 > q_f1)
            positive_vs_global += int(c_f1 > g_f1)
        gains_vs_qwen = np.asarray(gains_vs_qwen)
        gains_vs_global = np.asarray(gains_vs_global)
        records.append({
            "rank_full": rank,
            "weights": weights,
            "full_macro_f1": cand["full_macro_f1"],
            "full_gain_vs_qwen": cand["full_gain_vs_qwen"],
            "full_gain_vs_global": cand["full_gain_vs_global"],
            "holdout_gain_vs_qwen_mean": float(gains_vs_qwen.mean()),
            "holdout_gain_vs_qwen_min": float(gains_vs_qwen.min()),
            "holdout_gain_vs_qwen_max": float(gains_vs_qwen.max()),
            "holdout_positive_vs_qwen": int(positive_vs_qwen),
            "holdout_gain_vs_global_mean": float(gains_vs_global.mean()),
            "holdout_gain_vs_global_min": float(gains_vs_global.min()),
            "holdout_gain_vs_global_max": float(gains_vs_global.max()),
            "holdout_positive_vs_global": int(positive_vs_global),
            "holdout_total": int(args.holdout_seeds),
        })

    records.sort(key=lambda r: (r["holdout_gain_vs_global_mean"], r["holdout_positive_vs_global"], r["full_macro_f1"]), reverse=True)

    print("\nTop gated candidates by holdout gain vs GLOBAL w=0.05:")
    for row in records[:15]:
        print(
            f"  weights={row['weights']} full={row['full_macro_f1']:.6f} "
            f"full_gain_vs_global={row['full_gain_vs_global']:+.6f} "
            f"holdout_gain_vs_global_mean={row['holdout_gain_vs_global_mean']:+.6f} "
            f"pos_vs_global={row['holdout_positive_vs_global']}/{row['holdout_total']} "
            f"min={row['holdout_gain_vs_global_min']:+.6f}"
        )

    best = records[0]
    report = {
        "baseline_qwen_macro_f1": float(base_f1),
        "global_weight": float(args.global_weight),
        "global_macro_f1": float(global_f1),
        "global_gain_vs_qwen": float(global_f1 - base_f1),
        "best_gated_by_holdout_vs_global": best,
        "qwen_language_summary": summarize_by_lang(y, base_pred, langs, n_classes),
        "global_language_summary": summarize_by_lang(y, global_p, langs, n_classes),
        "best_gated_language_summary": summarize_by_lang(y, gated_pred(qwen, tree, langs, best["weights"]), langs, n_classes),
        "all_checked": records,
    }
    out = args.output_dir / "gated_holdout_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")

    if best["holdout_gain_vs_global_mean"] > 0.0003 and best["holdout_positive_vs_global"] >= int(args.holdout_seeds * 0.70) and best["holdout_gain_vs_global_min"] > -0.0015:
        print("\nVerdict: GATED_WORTH_CONSIDERING")
        print(f"Recommended gated candidate: {best['weights']}")
    else:
        print("\nVerdict: USE_GLOBAL")
        print("Recommended submission setting: global tree weight 0.05")

if __name__ == "__main__":
    main()
