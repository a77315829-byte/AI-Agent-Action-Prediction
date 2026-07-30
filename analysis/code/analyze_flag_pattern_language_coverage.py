#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check whether FLAG_PATTERNS / trigger patterns are under-covering English prompts.

Run from project root:

python .\analyze_flag_pattern_language_coverage.py `
  --data .\data\train.jsonl `
  --feature-utils .\feature_utils_qwen_v4.py `
  --text-scope current `
  --output-dir .\model\flag_pattern_lang_coverage

Recommended second run, if FLAG_PATTERNS are applied over constructed segments:

python .\analyze_flag_pattern_language_coverage.py `
  --data .\data\train.jsonl `
  --feature-utils .\feature_utils_qwen_v4.py `
  --text-scope segments `
  --output-dir .\model\flag_pattern_lang_coverage_segments
"""

import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


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


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("feature_utils_qwen_v4_loaded", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def language_bucket(text: str) -> Tuple[str, Dict[str, float]]:
    hangul = len(_HANGUL_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    total_letters = hangul + latin
    ko_ratio = hangul / max(total_letters, 1)
    latin_ratio = latin / max(total_letters, 1)

    if hangul >= 3 and latin >= 10:
        bucket = "mixed"
    elif hangul >= 3:
        bucket = "ko"
    elif latin >= 10:
        bucket = "en"
    elif hangul > 0 and latin > 0:
        bucket = "mixed"
    elif hangul > 0:
        bucket = "ko"
    elif latin > 0:
        bucket = "en"
    else:
        bucket = "other"

    return bucket, {
        "hangul_chars": hangul,
        "latin_chars": latin,
        "ko_ratio": ko_ratio,
        "latin_ratio": latin_ratio,
    }


def get_current_text(sample: Dict[str, Any], module: Any = None) -> str:
    for key in ["current_prompt", "prompt", "instruction", "query"]:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value

    if module is not None and hasattr(module, "build_segments"):
        try:
            segments = module.build_segments(sample)
            value = segments.get("current", "")
            if isinstance(value, str) and value.strip():
                return value
        except Exception:
            pass

    return json.dumps(sample, ensure_ascii=False)


def get_segment_text(sample: Dict[str, Any], module: Any = None) -> str:
    if module is not None and hasattr(module, "build_segments"):
        try:
            segments = module.build_segments(sample)
            if isinstance(segments, dict):
                parts = []
                for key in ["current", "history", "action", "meta", "workspace"]:
                    value = segments.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(f"[{key}]\n{value}")
                if parts:
                    return "\n\n".join(parts)
        except Exception:
            pass

    return json.dumps(sample, ensure_ascii=False)


def get_text(sample: Dict[str, Any], module: Any, scope: str) -> str:
    if scope == "current":
        return get_current_text(sample, module)
    if scope == "segments":
        return get_segment_text(sample, module)
    if scope == "full":
        return json.dumps(sample, ensure_ascii=False)
    raise ValueError(scope)


def flatten_pattern_value(value: Any) -> List[str]:
    out = []
    if value is None:
        return out

    # compiled regex
    if hasattr(value, "pattern") and isinstance(getattr(value, "pattern"), str):
        return [value.pattern]

    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        for v in value.values():
            out.extend(flatten_pattern_value(v))
        return out

    if isinstance(value, (list, tuple, set)):
        for item in value:
            out.extend(flatten_pattern_value(item))
        return out

    return out


def normalize_patterns(obj: Any, prefix: str = "flag") -> List[Tuple[str, str]]:
    patterns = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            name = str(key)
            for pat in flatten_pattern_value(value):
                patterns.append((name, pat))
        return patterns

    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                # e.g. {"name": "...", "pattern": "..."} or {"flag": ["..."]}
                if "name" in item and ("pattern" in item or "regex" in item or "patterns" in item):
                    name = str(item["name"])
                    value = item.get("pattern", item.get("regex", item.get("patterns")))
                    for pat in flatten_pattern_value(value):
                        patterns.append((name, pat))
                else:
                    for key, value in item.items():
                        for pat in flatten_pattern_value(value):
                            patterns.append((str(key), pat))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                name = str(item[0])
                # If tuple is (name, pattern) this works. If it is just many strings, flatten still works.
                for pat in flatten_pattern_value(item[1:]):
                    patterns.append((name, pat))
            else:
                for pat in flatten_pattern_value(item):
                    patterns.append((f"{prefix}_{i:03d}", pat))
        return patterns

    for pat in flatten_pattern_value(obj):
        patterns.append((prefix, pat))
    return patterns


def find_pattern_objects(module: Any) -> Dict[str, Any]:
    candidates = {}
    for name in dir(module):
        upper = name.upper()
        if ("PATTERN" in upper or "TRIGGER" in upper or "KEYWORD" in upper or "FLAG" in upper) and not name.startswith("__"):
            value = getattr(module, name)
            if isinstance(value, (dict, list, tuple, set, str)) or hasattr(value, "pattern"):
                candidates[name] = value
    return candidates


def compile_patterns(patterns: List[Tuple[str, str]]) -> List[Tuple[str, str, re.Pattern]]:
    compiled = []
    seen = set()
    for name, pat in patterns:
        if not isinstance(pat, str):
            continue
        pat = pat.strip()
        if not pat:
            continue
        key = (name, pat)
        if key in seen:
            continue
        seen.add(key)

        # Prefer regex. If invalid, fall back to escaped literal.
        try:
            rx = re.compile(pat, flags=re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pat), flags=re.IGNORECASE)
        compiled.append((name, pat, rx))
    return compiled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--feature-utils", type=Path, default=Path("feature_utils_qwen_v4.py"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--text-scope", choices=["current", "segments", "full"], default="current")
    ap.add_argument("--pattern-object", type=str, default="FLAG_PATTERNS",
                    help="Preferred object name. If missing, script will scan PATTERN/TRIGGER/KEYWORD/FLAG variables.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    module = load_module(args.feature_utils)
    samples = load_jsonl(args.data)

    if hasattr(module, args.pattern_object):
        raw_objects = {args.pattern_object: getattr(module, args.pattern_object)}
    else:
        raw_objects = find_pattern_objects(module)

    all_patterns = []
    object_counts = {}
    for obj_name, obj in raw_objects.items():
        normalized = normalize_patterns(obj, prefix=obj_name)
        object_counts[obj_name] = len(normalized)
        for flag_name, pat in normalized:
            all_patterns.append((f"{obj_name}.{flag_name}", pat))

    compiled = compile_patterns(all_patterns)
    if not compiled:
        raise RuntimeError(
            f"No usable patterns found. Checked preferred object={args.pattern_object!r}. "
            f"Found candidate objects: {list(raw_objects.keys())}"
        )

    print("Pattern objects found:")
    for k, v in object_counts.items():
        print(f"  {k}: {v} raw patterns")
    print(f"Usable compiled patterns: {len(compiled)}")
    print(f"Text scope: {args.text_scope}")

    rows = []
    per_flag_counts = defaultdict(lambda: defaultdict(int))
    per_lang_counts = defaultdict(int)

    for i, sample in enumerate(samples):
        text = get_text(sample, module, args.text_scope)
        lang, stats = language_bucket(get_current_text(sample, module))
        per_lang_counts[lang] += 1

        matched_flags = set()
        matched_patterns = 0
        for flag_name, pat, rx in compiled:
            if rx.search(text):
                matched_flags.add(flag_name)
                matched_patterns += 1
                per_flag_counts[flag_name][lang] += 1

        rows.append({
            "row_index": i,
            "id": str(sample.get("id", "")),
            "lang": lang,
            "matched_flag_count": len(matched_flags),
            "matched_pattern_count": matched_patterns,
            "any_flag_match": len(matched_flags) > 0,
            **stats,
            "text_preview": text[:240].replace("\n", " "),
        })

    df = pd.DataFrame(rows)

    summary = (
        df.groupby("lang")
        .agg(
            n=("id", "count"),
            any_match_rate=("any_flag_match", "mean"),
            zero_match_rate=("any_flag_match", lambda x: 1.0 - float(np.mean(x))),
            avg_matched_flags=("matched_flag_count", "mean"),
            median_matched_flags=("matched_flag_count", "median"),
            avg_matched_patterns=("matched_pattern_count", "mean"),
            median_matched_patterns=("matched_pattern_count", "median"),
            avg_hangul_chars=("hangul_chars", "mean"),
            avg_latin_chars=("latin_chars", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )

    flag_rows = []
    langs = sorted(per_lang_counts.keys())
    for flag_name in sorted(per_flag_counts.keys()):
        row = {"flag": flag_name}
        for lang in langs:
            n = per_lang_counts[lang]
            cnt = per_flag_counts[flag_name].get(lang, 0)
            row[f"{lang}_count"] = cnt
            row[f"{lang}_rate"] = cnt / max(n, 1)
        if "en" in langs and "ko" in langs:
            row["ko_minus_en_rate"] = row.get("ko_rate", 0.0) - row.get("en_rate", 0.0)
            row["en_minus_ko_rate"] = row.get("en_rate", 0.0) - row.get("ko_rate", 0.0)
        if "en" in langs and "mixed" in langs:
            row["mixed_minus_en_rate"] = row.get("mixed_rate", 0.0) - row.get("en_rate", 0.0)
        flag_rows.append(row)

    flag_df = pd.DataFrame(flag_rows)
    if "ko_minus_en_rate" in flag_df.columns:
        en_deficit = flag_df.sort_values("ko_minus_en_rate", ascending=False)
    else:
        en_deficit = flag_df

    patterns_df = pd.DataFrame(
        [{"flag": name, "pattern": pat} for name, pat, _ in compiled]
    )

    print("\n=== Summary by language ===")
    print(summary.to_string(index=False))

    if "ko_minus_en_rate" in flag_df.columns:
        print("\n=== Top flags with lower English coverage than Korean ===")
        cols = ["flag", "ko_rate", "en_rate", "mixed_rate", "ko_minus_en_rate", "mixed_minus_en_rate"]
        cols = [c for c in cols if c in en_deficit.columns]
        print(en_deficit[cols].head(30).to_string(index=False))

        print("\n=== Top flags with higher English coverage than Korean ===")
        cols2 = ["flag", "en_rate", "ko_rate", "mixed_rate", "en_minus_ko_rate"]
        cols2 = [c for c in cols2 if c in flag_df.columns]
        print(flag_df.sort_values("en_minus_ko_rate", ascending=False)[cols2].head(30).to_string(index=False))

    df.to_csv(args.output_dir / "sample_flag_matches.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "summary_by_language.csv", index=False, encoding="utf-8-sig")
    flag_df.to_csv(args.output_dir / "flag_rate_by_language.csv", index=False, encoding="utf-8-sig")
    patterns_df.to_csv(args.output_dir / "extracted_patterns.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "pattern_objects": object_counts,
        "compiled_pattern_count": len(compiled),
        "text_scope": args.text_scope,
        "language_counts": dict(per_lang_counts),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
