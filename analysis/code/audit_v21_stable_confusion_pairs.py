import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score

try:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import StratifiedKFold
except Exception:
    ExtraTreesClassifier = None
    StratifiedKFold = None


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

ACTION_TO_ID = {name: i for i, name in enumerate(ALL_CLASSES)}
ACTION_TO_FAMILY = np.asarray([
    0, 0, 0, 0,
    1, 1, 1,
    2, 2, 2,
    3, 3,
    4,
    3,
], dtype=np.int64)

DEFAULT_PAIRS = [
    "ask_user->plan_task",
    "list_directory->grep_search",
    "list_directory->read_file",
    "run_bash->lint_or_typecheck",
    "read_file->list_directory",
    "grep_search->read_file",
    "lint_or_typecheck->run_bash",
]

TEXT_KEYS_PRIORITY = [
    "current",
    "current_instruction",
    "instruction",
    "user_request",
    "query",
    "prompt",
    "task",
    "message",
    "input",
    "history",
    "messages",
    "conversation",
    "events",
    "actions",
    "open_files",
    "file_context",
]


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0))


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def entropy(prob: np.ndarray) -> np.ndarray:
    return -np.sum(prob * np.log(np.maximum(prob, 1e-12)), axis=1)


def top_values(prob: np.ndarray, k: int = 5) -> np.ndarray:
    return np.sort(prob, axis=1)[:, -k:][:, ::-1]


def one_hot(idx: np.ndarray, num: int) -> np.ndarray:
    out = np.zeros((len(idx), num), dtype=np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return out


def final_logits(action_logits: np.ndarray, family_logits: Optional[np.ndarray], postprocess: dict, cfg: dict) -> np.ndarray:
    out = action_logits.astype(np.float64) / float(cfg.get("action_temperature", 1.0))
    family_weight = float(cfg.get("family_weight", 0.0))
    prior_beta = float(cfg.get("prior_beta", 0.0))

    if family_logits is not None and family_weight != 0:
        out = out + family_weight * family_logits.astype(np.float64)[:, ACTION_TO_FAMILY]

    if prior_beta != 0:
        class_weights = np.asarray(
            postprocess.get("training_class_weights", np.ones(len(ALL_CLASSES))),
            dtype=np.float64,
        )
        out = out - prior_beta * np.log(np.maximum(class_weights, 1e-12))[None, :]

    return out


def tune_model(y: np.ndarray, action_logits: np.ndarray, family_logits: Optional[np.ndarray], postprocess: dict, name: str) -> Tuple[np.ndarray, dict, float]:
    configs = [{"action_temperature": 1.0, "family_weight": 0.0, "prior_beta": 0.0}]

    for t in [0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0]:
        for fw in [0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.25]:
            for pb in [-1.5, -1.0, -0.75, -0.5, 0.0, 0.25, 0.5]:
                configs.append({"action_temperature": t, "family_weight": fw, "prior_beta": pb})

    best_score = -1.0
    best_logits = None
    best_cfg = None

    for cfg in configs:
        logits = final_logits(action_logits, family_logits, postprocess, cfg)
        pred = logits.argmax(axis=1)
        score = macro_f1(y, pred)
        if score > best_score:
            best_score = score
            best_logits = logits
            best_cfg = dict(cfg)

    print(f"{name} tuned Macro-F1: {best_score:.6f}")
    print(f"{name} cfg: {json.dumps(best_cfg, ensure_ascii=False)}")
    return best_logits, best_cfg, best_score


def build_features(q_logits: np.ndarray, e_logits: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    q_prob = softmax(q_logits)
    e_prob = softmax(e_logits)
    q_logp = log_softmax(q_logits)
    e_logp = log_softmax(e_logits)

    q_pred = q_prob.argmax(axis=1)
    e_pred = e_prob.argmax(axis=1)

    q_top = np.argsort(q_prob, axis=1)[:, ::-1][:, :5]
    e_top = np.argsort(e_prob, axis=1)[:, ::-1][:, :5]

    q_tv = top_values(q_prob, 5)
    e_tv = top_values(e_prob, 5)

    q_margin12 = q_tv[:, 0] - q_tv[:, 1]
    q_margin13 = q_tv[:, 0] - q_tv[:, 2]
    e_margin12 = e_tv[:, 0] - e_tv[:, 1]
    e_margin13 = e_tv[:, 0] - e_tv[:, 2]

    e_in_q_top3 = np.asarray([e_pred[i] in q_top[i, :3] for i in range(len(q_pred))], dtype=bool)
    q_in_e_top3 = np.asarray([q_pred[i] in e_top[i, :3] for i in range(len(q_pred))], dtype=bool)

    pair_idx = q_pred * len(ALL_CLASSES) + e_pred

    features = [
        q_prob.astype(np.float32),
        e_prob.astype(np.float32),
        q_logp.astype(np.float32),
        e_logp.astype(np.float32),
        (e_logp - q_logp).astype(np.float32),
        (e_prob - q_prob).astype(np.float32),
        q_tv.astype(np.float32),
        e_tv.astype(np.float32),
        np.stack([
            q_margin12,
            q_margin13,
            e_margin12,
            e_margin13,
            q_prob.max(axis=1),
            e_prob.max(axis=1),
            entropy(q_prob),
            entropy(e_prob),
            (q_pred == e_pred).astype(np.float32),
            e_in_q_top3.astype(np.float32),
            q_in_e_top3.astype(np.float32),
        ], axis=1).astype(np.float32),
        one_hot(q_pred, len(ALL_CLASSES)),
        one_hot(e_pred, len(ALL_CLASSES)),
        one_hot(pair_idx, len(ALL_CLASSES) * len(ALL_CLASSES)),
    ]

    meta = {
        "q_pred": q_pred,
        "e_pred": e_pred,
        "q_top": q_top,
        "e_top": e_top,
        "q_margin12": q_margin12,
        "e_margin12": e_margin12,
        "q_conf": q_prob.max(axis=1),
        "e_conf": e_prob.max(axis=1),
        "e_in_q_top3": e_in_q_top3,
        "q_in_e_top3": q_in_e_top3,
    }
    return np.concatenate(features, axis=1), meta


def fit_oof_accept_prob(X: np.ndarray, target: np.ndarray, seed: int) -> np.ndarray:
    if ExtraTreesClassifier is None or StratifiedKFold is None:
        print("WARNING: sklearn ExtraTrees unavailable. accept_prob will be zeros.")
        return np.zeros(len(target), dtype=np.float64)
    if int(target.sum()) == 0:
        return np.zeros(len(target), dtype=np.float64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_prob = np.zeros(len(target), dtype=np.float64)

    for fold, (tr, va) in enumerate(skf.split(X, target), 1):
        clf = ExtraTreesClassifier(
            n_estimators=800,
            max_depth=9,
            min_samples_leaf=8,
            min_samples_split=16,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed + fold,
            n_jobs=-1,
        )
        clf.fit(X[tr], target[tr])
        oof_prob[va] = clf.predict_proba(X[va])[:, 1]
    return oof_prob


def parse_pairs(raw_pairs: Sequence[str]) -> List[Tuple[int, int, str]]:
    pairs = []
    for raw in raw_pairs:
        s = raw.strip()
        if not s:
            continue
        if "->" not in s:
            raise ValueError(f"Invalid pair format: {raw}. Expected qwen_pred->e5_pred")
        left, right = [x.strip() for x in s.split("->", 1)]
        if left not in ACTION_TO_ID or right not in ACTION_TO_ID:
            raise ValueError(f"Unknown class in pair: {raw}")
        pairs.append((ACTION_TO_ID[left], ACTION_TO_ID[right], f"{left}->{right}"))
    return pairs


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append({"raw_line": line})
    return rows


def try_get_indices(npz: Any, n: int) -> Optional[np.ndarray]:
    candidates = [
        "indices",
        "idx",
        "row_indices",
        "val_indices",
        "valid_indices",
        "validation_indices",
        "fold_indices",
        "sample_indices",
        "ids",
    ]
    for key in candidates:
        if key in npz.files:
            arr = np.asarray(npz[key]).reshape(-1)
            if len(arr) == n and np.issubdtype(arr.dtype, np.integer):
                print(f"Using validation indices from npz key: {key}")
                return arr.astype(np.int64)
    return None


def load_indices_file(path: Optional[Path], n: int) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path).reshape(-1)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        arr = None
        for key in data.files:
            cand = np.asarray(data[key]).reshape(-1)
            if len(cand) == n and np.issubdtype(cand.dtype, np.integer):
                arr = cand
                break
        if arr is None:
            raise RuntimeError(f"No valid index array found in {path}")
    else:
        vals = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for part in line.replace(",", " ").split():
                    try:
                        vals.append(int(part))
                    except ValueError:
                        pass
        arr = np.asarray(vals, dtype=np.int64)
    if len(arr) != n:
        raise RuntimeError(f"Index length mismatch: got {len(arr)}, expected {n}")
    return arr.astype(np.int64)


def compact_text(value: Any, max_len: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, ensure_ascii=False)
        except Exception:
            s = str(value)
    s = " ".join(s.split())
    if len(s) > max_len:
        return s[:max_len] + " ..."
    return s


def extract_text(row: Optional[dict], max_len: int = 700) -> str:
    if not row:
        return ""
    chunks = []
    for key in TEXT_KEYS_PRIORITY:
        if key in row and row[key] not in [None, "", [], {}]:
            chunks.append(f"[{key}] {compact_text(row[key], max_len=250)}")
    if not chunks:
        return compact_text(row, max_len=max_len)
    out = " | ".join(chunks)
    if len(out) > max_len:
        out = out[:max_len] + " ..."
    return out


def full_json(row: Optional[dict], max_len: int = 4000) -> str:
    if row is None:
        return ""
    try:
        s = json.dumps(row, ensure_ascii=False)
    except Exception:
        s = str(row)
    if len(s) > max_len:
        s = s[:max_len] + " ..."
    return s


def bucket_name(q_ok: bool, e_ok: bool) -> str:
    if q_ok and not e_ok:
        return "qwen_correct_e5_wrong"
    if (not q_ok) and e_ok:
        return "qwen_wrong_e5_correct"
    if (not q_ok) and (not e_ok):
        return "both_wrong"
    return "both_correct_unexpected"


def choose_samples(indices: np.ndarray, score: np.ndarray, max_per_bucket: int, seed: int) -> np.ndarray:
    if len(indices) <= max_per_bucket:
        return indices
    # 사람이 볼 때 유용하게 selector 확신이 높은 것 절반, 랜덤 절반.
    order = indices[np.argsort(-score[indices])]
    top_k = max_per_bucket // 2
    chosen = list(order[:top_k])
    remaining = np.asarray([i for i in indices if i not in set(chosen)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    chosen.extend(remaining[: max_per_bucket - len(chosen)].tolist())
    return np.asarray(chosen, dtype=np.int64)


def write_markdown(rows: List[dict], path: Path, max_items: int = 500) -> None:
    lines = []
    lines.append("# Stable Confusion Pair Audit Samples")
    lines.append("")
    lines.append("이 파일은 사람이 직접 라벨/경계 문제를 확인하기 위한 샘플 목록입니다.")
    lines.append("")

    current_pair = None
    current_bucket = None
    count = 0
    for row in rows[:max_items]:
        if row["pair"] != current_pair:
            current_pair = row["pair"]
            current_bucket = None
            lines.append(f"\n## {current_pair}\n")
        if row["bucket"] != current_bucket:
            current_bucket = row["bucket"]
            lines.append(f"\n### {current_bucket}\n")
        count += 1
        lines.append(f"#### {count}. local_idx={row['local_idx']} global_idx={row['global_idx']} true={row['true_label']}")
        lines.append(f"- qwen={row['qwen_pred']} conf={row['q_conf']} margin={row['q_margin12']}")
        lines.append(f"- e5={row['e5_pred']} conf={row['e_conf']} margin={row['e_margin12']} accept_prob={row['accept_prob']}")
        lines.append(f"- text: {row['text']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-logits", type=Path, required=True)
    parser.add_argument("--encoder-logits", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None, help="train.jsonl. Optional but strongly recommended for text inspection.")
    parser.add_argument("--val-indices", type=Path, default=None, help="Optional validation row indices file if not stored in npz.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS)
    parser.add_argument("--max-per-bucket", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-selector", action="store_true", help="Skip ExtraTrees accept_prob recomputation.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))
    q_npz = np.load(args.qwen_logits)
    e_npz = np.load(args.encoder_logits)

    required = ["labels", "action_logits"]
    for key in required:
        if key not in q_npz.files:
            raise RuntimeError(f"Qwen npz missing key: {key}")
        if key not in e_npz.files:
            raise RuntimeError(f"E5 npz missing key: {key}")

    y = q_npz["labels"].astype(np.int64)
    e_y = e_npz["labels"].astype(np.int64)
    if not np.array_equal(y, e_y):
        raise RuntimeError("Qwen/E5 labels are not aligned.")

    q_logits, q_cfg, q_score = tune_model(
        y,
        q_npz["action_logits"],
        q_npz["family_logits"] if "family_logits" in q_npz.files else None,
        postprocess,
        "Qwen",
    )
    e_logits, e_cfg, e_score = tune_model(
        y,
        e_npz["action_logits"],
        e_npz["family_logits"] if "family_logits" in e_npz.files else None,
        postprocess,
        "E5",
    )

    X, meta = build_features(q_logits, e_logits)
    q_pred = meta["q_pred"]
    e_pred = meta["e_pred"]
    q_ok = q_pred == y
    e_ok = e_pred == y
    target_accept = ((~q_ok) & e_ok).astype(np.int64)

    if args.no_selector:
        accept_prob = np.zeros(len(y), dtype=np.float64)
    else:
        accept_prob = fit_oof_accept_prob(X, target_accept, args.seed)

    print("\nAligned diagnostics:")
    print(f"  Qwen score: {q_score:.6f}")
    print(f"  E5 score:   {e_score:.6f}")
    print(f"  Qwen wrong / E5 correct: {int(((~q_ok) & e_ok).sum())}")
    print(f"  Qwen correct / E5 wrong: {int((q_ok & (~e_ok)).sum())}")

    data_rows = None
    val_indices = None
    if args.data is not None:
        data_rows = load_jsonl(args.data)
        val_indices = load_indices_file(args.val_indices, len(y)) if args.val_indices else None
        if val_indices is None:
            val_indices = try_get_indices(q_npz, len(y))
        if val_indices is None and len(data_rows) == len(y):
            print("Data row count equals validation size. Mapping local_idx directly to data row.")
            val_indices = np.arange(len(y), dtype=np.int64)
        elif val_indices is None:
            print("WARNING: Could not find validation indices. Text fields will be empty unless data has validation-size rows.")

    pairs = parse_pairs(args.pairs)
    rows_out: List[dict] = []
    summary_rows: List[dict] = []

    for q_id, e_id, pair_label in pairs:
        pair_mask = (q_pred == q_id) & (e_pred == e_id)
        pair_idx = np.where(pair_mask)[0]
        if len(pair_idx) == 0:
            summary_rows.append({
                "pair": pair_label,
                "count": 0,
                "qwen_correct_e5_wrong": 0,
                "qwen_wrong_e5_correct": 0,
                "both_wrong": 0,
                "sample_net": 0,
            })
            continue

        buckets = {
            "qwen_correct_e5_wrong": pair_idx[q_ok[pair_idx] & (~e_ok[pair_idx])],
            "qwen_wrong_e5_correct": pair_idx[(~q_ok[pair_idx]) & e_ok[pair_idx]],
            "both_wrong": pair_idx[(~q_ok[pair_idx]) & (~e_ok[pair_idx])],
        }
        sample_net = len(buckets["qwen_wrong_e5_correct"]) - len(buckets["qwen_correct_e5_wrong"])
        summary_rows.append({
            "pair": pair_label,
            "count": int(len(pair_idx)),
            "qwen_correct_e5_wrong": int(len(buckets["qwen_correct_e5_wrong"])),
            "qwen_wrong_e5_correct": int(len(buckets["qwen_wrong_e5_correct"])),
            "both_wrong": int(len(buckets["both_wrong"])),
            "sample_net": int(sample_net),
        })

        for bucket, idxs in buckets.items():
            chosen = choose_samples(idxs, accept_prob, args.max_per_bucket, args.seed + q_id * 100 + e_id)
            for local_idx in chosen:
                global_idx = ""
                row = None
                if data_rows is not None and val_indices is not None:
                    gi = int(val_indices[int(local_idx)])
                    global_idx = gi
                    if 0 <= gi < len(data_rows):
                        row = data_rows[gi]
                out = {
                    "pair": pair_label,
                    "bucket": bucket,
                    "local_idx": int(local_idx),
                    "global_idx": global_idx,
                    "true_id": int(y[local_idx]),
                    "true_label": ALL_CLASSES[int(y[local_idx])],
                    "qwen_pred_id": int(q_pred[local_idx]),
                    "qwen_pred": ALL_CLASSES[int(q_pred[local_idx])],
                    "e5_pred_id": int(e_pred[local_idx]),
                    "e5_pred": ALL_CLASSES[int(e_pred[local_idx])],
                    "q_conf": f"{float(meta['q_conf'][local_idx]):.6f}",
                    "e_conf": f"{float(meta['e_conf'][local_idx]):.6f}",
                    "q_margin12": f"{float(meta['q_margin12'][local_idx]):.6f}",
                    "e_margin12": f"{float(meta['e_margin12'][local_idx]):.6f}",
                    "accept_prob": f"{float(accept_prob[local_idx]):.6f}",
                    "e_in_q_top3": bool(meta["e_in_q_top3"][local_idx]),
                    "q_in_e_top3": bool(meta["q_in_e_top3"][local_idx]),
                    "text": extract_text(row),
                    "json": full_json(row),
                }
                rows_out.append(out)

    summary_path = args.output_dir / "stable_pair_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "count", "qwen_correct_e5_wrong", "qwen_wrong_e5_correct", "both_wrong", "sample_net"])
        writer.writeheader()
        writer.writerows(summary_rows)

    samples_path = args.output_dir / "stable_pair_audit_samples.csv"
    if rows_out:
        with samples_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
    else:
        samples_path.write_text("", encoding="utf-8")

    md_path = args.output_dir / "stable_pair_audit_samples.md"
    write_markdown(rows_out, md_path)

    meta_path = args.output_dir / "audit_meta.json"
    meta_payload = {
        "qwen_score": q_score,
        "e5_score": e_score,
        "qwen_cfg": q_cfg,
        "e5_cfg": e_cfg,
        "pairs": args.pairs,
        "max_per_bucket": args.max_per_bucket,
        "seed": args.seed,
        "qwen_npz_keys": list(q_npz.files),
        "e5_npz_keys": list(e_npz.files),
        "has_data": data_rows is not None,
        "has_val_indices": val_indices is not None,
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nPair summary:")
    for r in summary_rows:
        print(
            f"  {r['pair']:34s} count={r['count']:4d} "
            f"q_ok/e_wrong={r['qwen_correct_e5_wrong']:4d} "
            f"q_wrong/e_ok={r['qwen_wrong_e5_correct']:4d} "
            f"both_wrong={r['both_wrong']:4d} net={r['sample_net']:+4d}"
        )

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {samples_path}")
    print(f"  {md_path}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
