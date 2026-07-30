"""
analyze_file_search_oracle.py

file/search 4-way (read_file / grep_search / list_directory / glob_pattern)
boundary oracle analysis, per the project's own Section 15/16 spec.

Computes, using leak-free V12 validation logits only (no retraining):
  1. Full 14-way confusion matrix
  2. Per-class precision/recall/F1
  3. file/search group sample counts (correct vs wrong)
  4. Top-2 / top-3 accuracy, overall and within file/search
  5. Among WRONG rows, how often the true label is in top-2 / top-3
  6. Eligibility-gated oracle gain (matches Section 16's proposed specialist
     trigger: V12 top1 in file/search group, OR >=2 of V12's top2 in the group)
  7. Margin-bucketed oracle gain (does the group's oracle gain concentrate in
     low-margin/uncertain rows, or is it spread out -- low-margin concentration
     is a better sign for a margin-gated specialist)
  8. Major transition counts within the group

Given the V30 lesson (local+holdout gain of +0.0014 -> public -0.0018), treat
any oracle gain found here as an UPPER BOUND on what's achievable, not a
promise -- an actual specialist will recover only a fraction of this, and
even that fraction needs to survive public, which nothing in this pipeline
can fully guarantee.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils_qwen_v4 import ALL_CLASSES, ACTION_TO_FAMILY

NUM_CLASSES = len(ALL_CLASSES)
LABEL2ID = {label: i for i, label in enumerate(ALL_CLASSES)}

FILE_SEARCH_GROUP = ["read_file", "grep_search", "list_directory", "glob_pattern"]
FILE_SEARCH_IDS = [LABEL2ID[c] for c in FILE_SEARCH_GROUP]
FILE_SEARCH_SET = set(FILE_SEARCH_IDS)


def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def macro_f1_from_confusion(labels: np.ndarray, predictions: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return f1_score(labels, predictions, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--val-logits", type=Path, required=True, help="validation_logits_v12.npz")
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.val_logits)
    action_logits = npz["action_logits"].astype(np.float64)
    family_logits = npz["family_logits"].astype(np.float64)
    labels = npz["labels"].astype(np.int64) if "labels" in npz else None

    if labels is None:
        # fall back to val indices + labels csv, matching earlier project convention
        val_indices = npz["validation_indices"]
        with args.labels_csv.open(encoding="utf-8", newline="") as f:
            label_map = {str(row["id"]): str(row["action"]) for row in csv.DictReader(f)}
        samples = []
        with args.data.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        labels = np.asarray(
            [LABEL2ID[label_map[str(samples[i]["id"])]] for i in val_indices], dtype=np.int64
        )

    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))
    family_index = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)
    class_weights = np.asarray(postprocess["training_class_weights"], dtype=np.float64)
    final_logits = (
        action_logits / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"]) * family_logits[:, family_index]
        - float(postprocess["prior_beta"]) * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )
    proba = stable_softmax(final_logits)
    sorted_ids = np.argsort(-proba, axis=1)
    top1 = sorted_ids[:, 0]
    top2 = sorted_ids[:, 1]
    top3 = sorted_ids[:, 2]

    n = len(labels)
    results = {}

    # 1. Confusion matrix (only report the file/search x file/search block + escapes, to stay readable)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(labels, top1):
        confusion[t, p] += 1

    print("=== file/search block of confusion matrix (rows=true, cols=predicted) ===")
    header = "true\\pred".ljust(16) + "".join(c[:10].rjust(12) for c in FILE_SEARCH_GROUP)
    print(header)
    for t_name, t_id in zip(FILE_SEARCH_GROUP, FILE_SEARCH_IDS):
        row = t_name.ljust(16) + "".join(str(confusion[t_id, p_id]).rjust(12) for p_id in FILE_SEARCH_IDS)
        print(row)

    # 2. per-class precision/recall/F1 for the group
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, top1, labels=np.arange(NUM_CLASSES), zero_division=0
    )
    print("\n=== file/search per-class metrics ===")
    for c_id in FILE_SEARCH_IDS:
        print(f"  {ALL_CLASSES[c_id]:18s} precision={precision[c_id]:.4f} recall={recall[c_id]:.4f} f1={f1[c_id]:.4f} support={support[c_id]}")

    overall_f1 = macro_f1_from_confusion(labels, top1)
    print(f"\nOverall Macro-F1 (baseline): {overall_f1:.6f}")
    results["overall_macro_f1"] = overall_f1

    # 3. group sample counts
    true_in_group = np.isin(labels, FILE_SEARCH_IDS)
    correct = top1 == labels
    print(f"\nfile/search true samples: {int(true_in_group.sum())}")
    print(f"  correct: {int((true_in_group & correct).sum())}")
    print(f"  wrong:   {int((true_in_group & ~correct).sum())}")
    results["file_search_true_count"] = int(true_in_group.sum())
    results["file_search_correct"] = int((true_in_group & correct).sum())
    results["file_search_wrong"] = int((true_in_group & ~correct).sum())

    # 4. top2/top3 accuracy overall and within group
    top2_hit = (labels == top1) | (labels == top2)
    top3_hit = (labels == top1) | (labels == top2) | (labels == top3)
    print(f"\nOverall top1 acc: {correct.mean():.4f}  top2 acc: {top2_hit.mean():.4f}  top3 acc: {top3_hit.mean():.4f}")
    print(f"file/search top1 acc: {correct[true_in_group].mean():.4f}  top2 acc: {top2_hit[true_in_group].mean():.4f}  top3 acc: {top3_hit[true_in_group].mean():.4f}")
    results["overall_top1"] = float(correct.mean())
    results["overall_top2"] = float(top2_hit.mean())
    results["overall_top3"] = float(top3_hit.mean())
    results["file_search_top1"] = float(correct[true_in_group].mean())
    results["file_search_top2"] = float(top2_hit[true_in_group].mean())
    results["file_search_top3"] = float(top3_hit[true_in_group].mean())

    # 5. among WRONG file/search rows, is true label in top2/top3?
    wrong_group_mask = true_in_group & ~correct
    if wrong_group_mask.any():
        wrong_top2_rate = top2_hit[wrong_group_mask].mean()
        wrong_top3_rate = top3_hit[wrong_group_mask].mean()
        print(f"\nAmong WRONG file/search rows ({int(wrong_group_mask.sum())}):")
        print(f"  true label in top2: {wrong_top2_rate:.4f}")
        print(f"  true label in top3: {wrong_top3_rate:.4f}")
        results["wrong_group_top2_rate"] = float(wrong_top2_rate)
        results["wrong_group_top3_rate"] = float(wrong_top3_rate)

    # 6. eligibility-gated oracle (Section 16 trigger: top1 in group, OR >=2 of top2 in group)
    top1_in_group = np.isin(top1, FILE_SEARCH_IDS)
    top2_in_group_count = np.isin(top1, FILE_SEARCH_IDS).astype(int) + np.isin(top2, FILE_SEARCH_IDS).astype(int)
    eligible = top1_in_group | (top2_in_group_count >= 2)

    oracle_pred = top1.copy()
    # Oracle: on eligible rows, if the true label is in file/search group, use it (perfect specialist);
    # otherwise keep V12's original prediction (specialist wouldn't fire outside its 4 classes).
    use_oracle = eligible & true_in_group
    oracle_pred[use_oracle] = labels[use_oracle]

    oracle_f1 = macro_f1_from_confusion(labels, oracle_pred)
    print(f"\n=== Eligibility-gated oracle (top1 in group, or top2 has >=2 group members) ===")
    print(f"Eligible rows: {int(eligible.sum())} / {n} ({100*eligible.mean():.2f}%)")
    print(f"Oracle Macro-F1: {oracle_f1:.6f}  (gain vs baseline: {oracle_f1 - overall_f1:+.6f})")
    results["eligible_rows"] = int(eligible.sum())
    results["gated_oracle_macro_f1"] = oracle_f1
    results["gated_oracle_gain"] = oracle_f1 - overall_f1

    # 7. margin-bucketed oracle gain (within eligible rows)
    sorted_proba = np.sort(proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    buckets = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]
    print(f"\n=== Margin-bucketed oracle gain (eligible rows only) ===")
    bucket_results = []
    for lo, hi in buckets:
        bucket_mask = eligible & (margin >= lo) & (margin < hi)
        if bucket_mask.sum() == 0:
            continue
        bucket_wrong = bucket_mask & ~correct & true_in_group
        n_bucket = int(bucket_mask.sum())
        n_recoverable = int(bucket_wrong.sum())
        print(f"  margin [{lo:.1f},{hi:.1f}): n={n_bucket:5d}  recoverable(true in group & currently wrong)={n_recoverable:4d}  rate={n_recoverable/max(1,n_bucket):.4f}")
        bucket_results.append({"margin_lo": lo, "margin_hi": hi, "n": n_bucket, "recoverable": n_recoverable})
    results["margin_buckets"] = bucket_results

    # 8. major transitions within group
    print(f"\n=== Major transitions within file/search (true -> predicted, wrong only) ===")
    transition_counts = {}
    for t, p in zip(labels[wrong_group_mask], top1[wrong_group_mask]):
        key = (ALL_CLASSES[t], ALL_CLASSES[p])
        transition_counts[key] = transition_counts.get(key, 0) + 1
    for (t_name, p_name), count in sorted(transition_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {t_name:18s} -> {p_name:18s} : {count}")
    results["transitions"] = {f"{t}->{p}": c for (t, p), c in transition_counts.items()}

    # Verdict per the project's own thresholds (Section 15)
    gain = results["gated_oracle_gain"]
    top3_recover_rate = results.get("wrong_group_top3_rate", 0.0)
    if gain >= 0.005:
        verdict = "oracle gain >= +0.005 -> specialist 진행 가치 높음 (문서 기준)"
    elif gain >= 0.002:
        verdict = "oracle gain +0.002~+0.005 -> 애매, V30 사례 감안하면 신중하게"
    else:
        verdict = "oracle gain < +0.002 -> file/search도 기대치 낮음, ask_user/plan_task로 이동 권장"
    print(f"\nVerdict: {verdict}")
    print(f"(참고: top3 recovery rate={top3_recover_rate:.2%} -- 이게 낮으면 reranker를 만들어도 애초에 후보군에 정답이 없는 케이스가 많다는 뜻)")
    results["verdict"] = verdict

    if args.report:
        (args.output_dir / "file_search_oracle_report.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nSaved report to {args.output_dir / 'file_search_oracle_report.json'}")


if __name__ == "__main__":
    main()
