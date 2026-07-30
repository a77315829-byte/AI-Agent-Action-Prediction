"""
screen_viterbi_transition.py

Session-level Viterbi transition decoding, screened with leak-free K-fold
(transition matrix fit only on OTHER folds' true labels; emission logits
come from an already-OOF Qwen model, e.g. cur160). No retraining -- purely
a post-processing decode step over existing logits, applied per-session
using the "session_id-step_N" id structure.

score[t][action] = emission_logprob[t][action] + lambda * log(P(action | prev_action))

This is a genuinely different information source from everything else tried
in this project (all prior methods treated each row independently).
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils_qwen_v4 import ALL_CLASSES, ACTION_TO_FAMILY

NUM_CLASSES = len(ALL_CLASSES)
LABEL2ID = {label: i for i, label in enumerate(ALL_CLASSES)}


def load_jsonl(path: Path) -> List[dict]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def macro_f1(labels, predictions) -> float:
    return f1_score(labels, predictions, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0)


def parse_session_step(sample_id: str) -> Tuple[str, int]:
    session, _, step_part = sample_id.rpartition("-step_")
    try:
        step = int(step_part)
    except ValueError:
        step = 0
    return session, step


def build_sessions(ids: List[str]) -> Dict[str, List[int]]:
    """Returns {session_id: [row_indices sorted by step]}."""
    parsed = [(parse_session_step(sid), i) for i, sid in enumerate(ids)]
    sessions: Dict[str, List[Tuple[int, int]]] = {}
    for (session, step), idx in parsed:
        sessions.setdefault(session, []).append((step, idx))
    return {session: [idx for _, idx in sorted(steps)] for session, steps in sessions.items()}


def fit_transition_matrix(labels: np.ndarray, sessions: Dict[str, List[int]], smoothing: float) -> np.ndarray:
    counts = np.full((NUM_CLASSES, NUM_CLASSES), smoothing, dtype=np.float64)
    for row_indices in sessions.values():
        for prev_idx, curr_idx in zip(row_indices[:-1], row_indices[1:]):
            counts[labels[prev_idx], labels[curr_idx]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


def viterbi_decode_session(emission_logprob: np.ndarray, log_trans: np.ndarray, lam: float) -> np.ndarray:
    """emission_logprob: (T, NUM_CLASSES). Returns decoded path (T,)."""
    T = emission_logprob.shape[0]
    if T == 1:
        return emission_logprob.argmax(axis=1)

    value = np.zeros((T, NUM_CLASSES), dtype=np.float64)
    backptr = np.zeros((T, NUM_CLASSES), dtype=np.int64)
    value[0] = emission_logprob[0]

    for t in range(1, T):
        # value[t][s] = emission[t][s] + max_prev(value[t-1][prev] + lam * log_trans[prev, s])
        candidates = value[t - 1][:, None] + lam * log_trans  # (prev, curr)
        backptr[t] = candidates.argmax(axis=0)
        value[t] = emission_logprob[t] + candidates.max(axis=0)

    path = np.zeros(T, dtype=np.int64)
    path[-1] = value[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path


def decode_all(sessions: Dict[str, List[int]], proba: np.ndarray, log_trans: np.ndarray, lam: float) -> np.ndarray:
    n = proba.shape[0]
    predictions = np.zeros(n, dtype=np.int64)
    log_proba = np.log(np.maximum(proba, 1e-12))
    for row_indices in sessions.values():
        emission = log_proba[row_indices]
        path = viterbi_decode_session(emission, log_trans, lam)
        for local_i, row_idx in enumerate(row_indices):
            predictions[row_idx] = path[local_i]
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--qwen-oof", type=Path, required=True, help="e.g. model/oof_cur160/oof_logits_all_70000.npz")
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--smoothing-grid", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--tree-prob", type=Path, default=None,
                         help="if given, blend this leak-free tree OOF (N,14) into Qwen proba at --tree-weight "
                              "BEFORE Viterbi decoding, to match the actual deployment pipeline (Qwen+tree, not Qwen alone)")
    parser.add_argument("--tree-weight", type=float, default=0.15)
    parser.add_argument("--n-splits", type=int, default=5, help="K-fold for leak-free transition-matrix fitting")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Load data...")
    samples = load_jsonl(args.data)
    ids = [str(s["id"]) for s in samples]
    with args.labels_csv.open(encoding="utf-8", newline="") as f:
        label_map = {str(row["id"]): str(row["action"]) for row in csv.DictReader(f)}
    labels = np.asarray([LABEL2ID[label_map[i]] for i in ids], dtype=np.int64)
    groups = np.asarray([parse_session_step(i)[0] for i in ids])

    print("Load Qwen OOF...")
    oof = np.load(args.qwen_oof)
    action_logits = oof["action_logits"].astype(np.float64)
    family_logits = oof["family_logits"].astype(np.float64)
    oof_labels = oof["labels"].astype(np.int64)
    if not np.array_equal(labels, oof_labels):
        raise RuntimeError("OOF labels do not match labels file -- check row order.")

    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))
    family_index = np.asarray(ACTION_TO_FAMILY, dtype=np.int64)
    class_weights = np.asarray(postprocess["training_class_weights"], dtype=np.float64)
    final_logits = (
        action_logits / float(postprocess["action_temperature"])
        + float(postprocess["family_weight"]) * family_logits[:, family_index]
        - float(postprocess["prior_beta"]) * np.log(np.maximum(class_weights, 1e-12))[None, :]
    )
    qwen_proba = stable_softmax(final_logits)

    if args.tree_prob is not None:
        tree_proba = np.load(args.tree_prob)
        assert tree_proba.shape[0] == len(labels), f"tree_prob rows={tree_proba.shape[0]} vs labels={len(labels)}"
        w = args.tree_weight
        blended_log = (1 - w) * np.log(np.maximum(qwen_proba, 1e-12)) + w * np.log(np.maximum(tree_proba, 1e-12))
        qwen_proba = stable_softmax(blended_log)
        print(f"Blended Qwen+tree at weight={w} BEFORE Viterbi (matches production pipeline)")

    qwen_pred = qwen_proba.argmax(axis=1)
    qwen_f1 = macro_f1(labels, qwen_pred)
    print(f"Qwen baseline (argmax, no transition) Macro-F1: {qwen_f1:.6f}\n")

    session_lengths = [len(v) for v in build_sessions(ids).values()]
    print(f"Sessions: {len(session_lengths)}, step-length distribution: "
          f"min={min(session_lengths)} max={max(session_lengths)} mean={np.mean(session_lengths):.2f}\n")

    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    fold_indices = list(splitter.split(np.zeros(len(labels)), labels, groups))

    print("=== Leak-free K-fold screen (transition matrix fit on OTHER folds only) ===")
    best_overall = None
    results_by_config = []

    for smoothing in args.smoothing_grid:
        for lam in args.lambda_grid:
            fold_gains = []
            for eval_fold_idx, (train_idx, eval_idx) in enumerate(fold_indices):
                # transition matrix from TRAIN portion's true labels only
                eval_ids = [ids[i] for i in eval_idx]
                eval_sessions_local = build_sessions(eval_ids)
                # remap local session indices (0..len(eval_idx)-1) back to global row indices
                eval_sessions_global = {
                    session: [eval_idx[local_i] for local_i in local_indices]
                    for session, local_indices in eval_sessions_local.items()
                }

                train_ids = [ids[i] for i in train_idx]
                train_sessions_local = build_sessions(train_ids)
                train_sessions_global = {
                    session: [train_idx[local_i] for local_i in local_indices]
                    for session, local_indices in train_sessions_local.items()
                }
                log_trans = np.log(np.maximum(fit_transition_matrix(labels, train_sessions_global, smoothing), 1e-12))

                decoded = decode_all(eval_sessions_global, qwen_proba, log_trans, lam)
                argmax_pred_eval = qwen_pred[eval_idx]
                decoded_eval = decoded[eval_idx]

                argmax_f1 = macro_f1(labels[eval_idx], argmax_pred_eval)
                decoded_f1 = macro_f1(labels[eval_idx], decoded_eval)
                fold_gains.append(decoded_f1 - argmax_f1)

            fold_gains = np.asarray(fold_gains)
            positive = int((fold_gains > 0).sum())
            mean_gain = float(fold_gains.mean())
            results_by_config.append({
                "smoothing": smoothing, "lambda": lam,
                "gain_mean": mean_gain, "gain_min": float(fold_gains.min()),
                "positive": positive, "total": len(fold_gains),
            })
            if lam > 0:  # lambda=0 is the no-op baseline, don't let it "win"
                if best_overall is None or mean_gain > best_overall["gain_mean"]:
                    best_overall = results_by_config[-1]
            print(f"  smoothing={smoothing:.2f} lambda={lam:.2f}  "
                  f"gain_mean={mean_gain:+.6f}  gain_min={fold_gains.min():+.6f}  positive={positive}/{len(fold_gains)}")

    print()
    if best_overall is not None:
        print(f"Best config: smoothing={best_overall['smoothing']}, lambda={best_overall['lambda']}, "
              f"gain_mean={best_overall['gain_mean']:+.6f}, positive={best_overall['positive']}/{best_overall['total']}")

        positive_rate = best_overall["positive"] / best_overall["total"]
        if positive_rate >= 0.8 and best_overall["gain_mean"] > 0.001:
            verdict = "PROMISING - worth adding to script.py as a post-processing step, but confirm on a fresh public submission given recent local-vs-public gaps"
        elif positive_rate >= 0.6:
            verdict = "MARGINAL"
        else:
            verdict = "DISCARD"
        print(f"Verdict: {verdict}")

        if args.report:
            (args.output_dir / "viterbi_screen_report.json").write_text(
                json.dumps({"qwen_f1": qwen_f1, "best": best_overall, "all_configs": results_by_config, "verdict": verdict},
                           indent=2), encoding="utf-8"
            )
            print(f"Saved report to {args.output_dir}")


if __name__ == "__main__":
    main()
