import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit


ALL_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]

ACTION_TO_FAMILY = np.asarray([
    0, 0, 0, 0,
    1, 1, 1,
    2, 2, 2,
    3, 3,
    4,
    3,
], dtype=np.int64)


def macro_f1(y, pred):
    return float(f1_score(y, pred, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0))


def class_f1(y, pred):
    return f1_score(y, pred, labels=np.arange(len(ALL_CLASSES)), average=None, zero_division=0)


def softmax(logits):
    x = logits.astype(np.float64)
    x -= x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def log_softmax(logits):
    return np.log(np.maximum(softmax(logits), 1e-12))


def final_logits(action_logits, family_logits, postprocess, cfg):
    out = action_logits.astype(np.float64) / float(cfg.get("action_temperature", 1.0))
    fw = float(cfg.get("family_weight", 0.0))
    pb = float(cfg.get("prior_beta", 0.0))

    if family_logits is not None and fw != 0:
        out = out + fw * family_logits.astype(np.float64)[:, ACTION_TO_FAMILY]

    if pb != 0:
        class_weights = np.asarray(
            postprocess.get("training_class_weights", np.ones(len(ALL_CLASSES))),
            dtype=np.float64,
        )
        out = out - pb * np.log(np.maximum(class_weights, 1e-12))[None, :]

    return out


def tune_qwen(y, q_npz, postprocess):
    action_logits = q_npz["action_logits"]
    family_logits = q_npz["family_logits"] if "family_logits" in q_npz.files else None
    configs = [{"action_temperature": 1.0, "family_weight": 0.0, "prior_beta": 0.0}]

    for t in [0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0]:
        for fw in [0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.25]:
            for pb in [-1.5, -1.0, -0.75, -0.5, 0.0, 0.25, 0.5]:
                configs.append({"action_temperature": t, "family_weight": fw, "prior_beta": pb})

    best = None
    for cfg in configs:
        logits = final_logits(action_logits, family_logits, postprocess, cfg)
        pred = logits.argmax(axis=1)
        score = macro_f1(y, pred)
        if best is None or score > best["score"]:
            best = {"score": score, "cfg": cfg, "logits": logits, "pred": pred}

    return best


def write_csv(rows, path):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, key):
    arr = np.asarray([r[key] for r in rows], dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "positive": int((arr > 0).sum()),
        "n": int(len(arr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-logits", type=Path, required=True)
    ap.add_argument("--postprocess", type=Path, required=True)
    ap.add_argument("--tree-prob", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("model/v23b_fixed_weight_probe"))
    ap.add_argument("--weights", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40")
    ap.add_argument("--holdout-ratio", type=float, default=0.35)
    ap.add_argument("--seeds", default="11,22,33,42,55,66,77,88,99,123")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    q_npz = np.load(args.qwen_logits, allow_pickle=False)
    if "labels" not in q_npz.files:
        raise RuntimeError(f"qwen logits has no labels key. keys={q_npz.files}")
    y = q_npz["labels"].astype(np.int64)

    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))
    q = tune_qwen(y, q_npz, postprocess)

    tree_prob = np.load(args.tree_prob).astype(np.float64)
    if tree_prob.shape != (len(y), len(ALL_CLASSES)):
        raise RuntimeError(f"tree_prob shape mismatch: {tree_prob.shape}")

    q_lp = log_softmax(q["logits"])
    t_lp = np.log(np.maximum(tree_prob, 1e-12))

    weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    base_score = macro_f1(y, q["pred"])
    print(f"Qwen Macro-F1: {base_score:.6f}")
    print(f"Qwen cfg: {json.dumps(q['cfg'], ensure_ascii=False)}")

    full_rows = []
    seed_rows = []

    for w in weights:
        pred = ((1.0 - w) * q_lp + w * t_lp).argmax(axis=1)
        score = macro_f1(y, pred)

        q_f1 = class_f1(y, q["pred"])
        b_f1 = class_f1(y, pred)

        full_rows.append({
            "weight_tree": float(w),
            "full_score": float(score),
            "full_gain": float(score - base_score),
            **{f"{cls}_delta": float(b_f1[i] - q_f1[i]) for i, cls in enumerate(ALL_CLASSES)},
        })

        for seed in seeds:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.holdout_ratio, random_state=seed)
            tr, ho = next(splitter.split(np.zeros(len(y)), y))

            q_tr = macro_f1(y[tr], q["pred"][tr])
            q_ho = macro_f1(y[ho], q["pred"][ho])
            b_tr = macro_f1(y[tr], pred[tr])
            b_ho = macro_f1(y[ho], pred[ho])

            seed_rows.append({
                "weight_tree": float(w),
                "seed": int(seed),
                "train_qwen": float(q_tr),
                "train_blend": float(b_tr),
                "train_gain": float(b_tr - q_tr),
                "holdout_qwen": float(q_ho),
                "holdout_blend": float(b_ho),
                "holdout_gain": float(b_ho - q_ho),
                "all_qwen": float(base_score),
                "all_blend": float(score),
                "all_gain": float(score - base_score),
            })

    summary_rows = []
    for w in weights:
        rows = [r for r in seed_rows if abs(r["weight_tree"] - w) < 1e-12]
        train_s = summarize(rows, "train_gain")
        hold_s = summarize(rows, "holdout_gain")
        all_s = summarize(rows, "all_gain")
        full = [r for r in full_rows if abs(r["weight_tree"] - w) < 1e-12][0]
        summary_rows.append({
            "weight_tree": float(w),
            "full_score": full["full_score"],
            "full_gain": full["full_gain"],
            "train_gain_mean": train_s["mean"],
            "train_gain_min": train_s["min"],
            "train_gain_max": train_s["max"],
            "train_gain_positive": train_s["positive"],
            "holdout_gain_mean": hold_s["mean"],
            "holdout_gain_min": hold_s["min"],
            "holdout_gain_max": hold_s["max"],
            "holdout_gain_positive": hold_s["positive"],
            "all_gain_mean": all_s["mean"],
            "n": hold_s["n"],
        })

    summary_rows.sort(key=lambda r: (r["holdout_gain_mean"], r["full_gain"]), reverse=True)

    print()
    print("Fixed-weight stability summary:")
    for r in summary_rows:
        print(
            f"  w={r['weight_tree']:.2f} "
            f"full_gain={r['full_gain']:+.6f} "
            f"holdout_mean={r['holdout_gain_mean']:+.6f} "
            f"holdout_min={r['holdout_gain_min']:+.6f} "
            f"positive={r['holdout_gain_positive']}/{r['n']}"
        )

    best = summary_rows[0]
    best_w = best["weight_tree"]
    best_pred = ((1.0 - best_w) * q_lp + best_w * t_lp).argmax(axis=1)
    np.save(args.output_dir / "pred_fixed_best.npy", best_pred.astype(np.int64))

    if args.report:
        print()
        print(f"Best fixed weight: {best_w:.2f}")
        print(f"Best fixed score:  {macro_f1(y, best_pred):.6f}")
        print()
        print("Class F1 Qwen -> Fixed blend:")
        q_f1 = class_f1(y, q["pred"])
        b_f1 = class_f1(y, best_pred)
        for i, cls in enumerate(ALL_CLASSES):
            print(f"{cls:18s} {q_f1[i]:.6f} -> {b_f1[i]:.6f} ({b_f1[i]-q_f1[i]:+.6f})")

    write_csv(full_rows, args.output_dir / "fixed_weight_full_scores.csv")
    write_csv(seed_rows, args.output_dir / "fixed_weight_seed_scores.csv")
    write_csv(summary_rows, args.output_dir / "fixed_weight_summary.csv")

    summary = {
        "qwen_score": base_score,
        "qwen_cfg": q["cfg"],
        "best_fixed_weight": best_w,
        "best_fixed_summary": best,
        "all_weight_summary": summary_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print(f"  {args.output_dir / 'summary.json'}")
    print(f"  {args.output_dir / 'fixed_weight_summary.csv'}")
    print(f"  {args.output_dir / 'fixed_weight_seed_scores.csv'}")

    if best["holdout_gain_mean"] > 0.0005 and best["holdout_gain_positive"] >= 7:
        print()
        print("Decision hint: fixed tree blend is stable enough for submit candidate integration.")
    else:
        print()
        print("Decision hint: fixed tree blend is weaker than tuned-weight result. Consider constrained selector or lower-risk submit only.")


if __name__ == "__main__":
    main()
