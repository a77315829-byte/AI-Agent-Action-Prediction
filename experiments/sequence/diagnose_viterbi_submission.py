from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


PREDICTION_PATTERN = re.compile(
    r'''
    (?P<indent>^[ \t]+)
    predictions[ \t]*=[ \t]*viterbi_decode_all\(
        [ \t\r\n]*ids[ \t]*,
        [ \t\r\n]*blended_log_probability[ \t]*,
        [ \t\r\n]*transition_matrix[ \t]*,
        [ \t\r\n]*viterbi_lambda[ \t]*,
        [ \t\r\n]*\)
    ''',
    re.MULTILINE | re.VERBOSE,
)


def patch_script(script_path: Path) -> None:
    text = script_path.read_text(encoding="utf-8")

    if "VITERBI_DIAGNOSTIC_JSON:" in text:
        print("Diagnostic code already exists:", script_path)
        return

    match = PREDICTION_PATTERN.search(text)
    if match is None:
        raise RuntimeError(
            "Could not find the Viterbi prediction block in script.py. "
            "Confirm that this is the submitted Viterbi ZIP."
        )

    indent = match.group("indent")
    replacement = (
        f"{indent}baseline_predictions = blended_log_probability.argmax(axis=1)\n\n"
        f"{indent}predictions = viterbi_decode_all(\n"
        f"{indent}    ids,\n"
        f"{indent}    blended_log_probability,\n"
        f"{indent}    transition_matrix,\n"
        f"{indent}    viterbi_lambda,\n"
        f"{indent})\n\n"
        f"{indent}changed_mask = predictions != baseline_predictions\n"
        f"{indent}sessions_diag = build_sessions(ids)\n"
        f"{indent}session_lengths_diag = [\n"
        f"{indent}    len(indices)\n"
        f"{indent}    for indices in sessions_diag.values()\n"
        f"{indent}]\n"
        f"{indent}diagnostic = {{\n"
        f"{indent}    'changed_rows': int(changed_mask.sum()),\n"
        f"{indent}    'total_rows': int(len(predictions)),\n"
        f"{indent}    'sessions': int(len(session_lengths_diag)),\n"
        f"{indent}    'multi_step_sessions': int(\n"
        f"{indent}        sum(length > 1 for length in session_lengths_diag)\n"
        f"{indent}    ),\n"
        f"{indent}    'max_session_length': int(\n"
        f"{indent}        max(session_lengths_diag)\n"
        f"{indent}        if session_lengths_diag else 0\n"
        f"{indent}    ),\n"
        f"{indent}    'viterbi_lambda': float(viterbi_lambda),\n"
        f"{indent}    'tree_weight': float(tree_blend_weight),\n"
        f"{indent}}}\n"
        f"{indent}print(\n"
        f"{indent}    'VITERBI_DIAGNOSTIC_JSON:',\n"
        f"{indent}    json.dumps(diagnostic, ensure_ascii=False),\n"
        f"{indent})\n\n"
        f"{indent}if changed_mask.any():\n"
        f"{indent}    changed_pairs = {{}}\n"
        f"{indent}    for before, after in zip(\n"
        f"{indent}        baseline_predictions[changed_mask],\n"
        f"{indent}        predictions[changed_mask],\n"
        f"{indent}    ):\n"
        f"{indent}        key = (\n"
        f"{indent}            ALL_CLASSES[int(before)],\n"
        f"{indent}            ALL_CLASSES[int(after)],\n"
        f"{indent}        )\n"
        f"{indent}        changed_pairs[key] = changed_pairs.get(key, 0) + 1\n\n"
        f"{indent}    print('Top Viterbi prediction changes:')\n"
        f"{indent}    for (before_name, after_name), count in sorted(\n"
        f"{indent}        changed_pairs.items(),\n"
        f"{indent}        key=lambda item: item[1],\n"
        f"{indent}        reverse=True,\n"
        f"{indent}    )[:20]:\n"
        f"{indent}        print(f'  {{before_name}} -> {{after_name}}: {{count}}')"
    )

    patched = text[: match.start()] + replacement + text[match.end() :]
    script_path.write_text(patched, encoding="utf-8")
    print("Patched diagnostic code:", script_path)


def validate_zip(zip_path: Path) -> None:
    required = {
        "script.py",
        "model/feature_utils_qwen_v4.py",
        "model/tree_blend_config.json",
        "model/transition_matrix.npy",
        "model/viterbi_config.json",
    }

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = {
            name.replace("\\\\", "/").lstrip("./")
            for name in archive.namelist()
        }

    missing = sorted(required - names)
    if missing:
        raise RuntimeError(
            "Submission ZIP is missing required files: "
            + ", ".join(missing)
        )


def run_submission(
    work_dir: Path,
    data_dir: Path,
    output_dir: Path,
    batch_size: int | None,
    log_path: Path,
) -> int:
    env = os.environ.copy()
    env["DACON_DATA_DIR"] = str(data_dir.resolve())
    env["DACON_OUTPUT_DIR"] = str(output_dir.resolve())
    env["PYTHONUNBUFFERED"] = "1"
    if batch_size is not None:
        env["BATCH_SIZE"] = str(batch_size)

    command = [sys.executable, "-u", str(work_dir / "script.py")]
    print("Run:", " ".join(command))
    print("DACON_DATA_DIR:", env["DACON_DATA_DIR"])
    print("DACON_OUTPUT_DIR:", env["DACON_OUTPUT_DIR"])

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(work_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return int(process.wait())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--work-dir", type=Path, default=Path("diag_submit_viterbi"))
    parser.add_argument("--output-dir", type=Path, default=Path("diag_viterbi_output"))
    parser.add_argument("--log", type=Path, default=Path("diag_viterbi.log"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    zip_path = args.zip.resolve()
    data_dir = args.data_dir.resolve()
    work_dir = args.work_dir.resolve()
    output_dir = args.output_dir.resolve()
    log_path = args.log.resolve()

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    if not (data_dir / "test.jsonl").exists():
        raise FileNotFoundError(data_dir / "test.jsonl")
    if not (data_dir / "sample_submission.csv").exists():
        raise FileNotFoundError(data_dir / "sample_submission.csv")

    validate_zip(zip_path)

    if not args.keep_existing:
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Extract:", zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(work_dir)

    patch_script(work_dir / "script.py")

    exit_code = run_submission(
        work_dir=work_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        log_path=log_path,
    )

    print()
    print("Process exit code:", exit_code)
    print("Log:", log_path)
    print("Output:", output_dir / "submission.csv")

    if exit_code != 0:
        raise SystemExit(exit_code)

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"VITERBI_DIAGNOSTIC_JSON:\s*(\{.*\})", log_text)
    if match is None:
        raise RuntimeError("The run finished, but diagnostic JSON was not found.")

    diagnostic = json.loads(match.group(1))
    print()
    print("=== Viterbi diagnostic summary ===")
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))

    changed_rows = int(diagnostic["changed_rows"])
    multi_step_sessions = int(diagnostic["multi_step_sessions"])

    if multi_step_sessions == 0:
        print("Verdict: SESSION PARSING FAILED")
    elif changed_rows == 0:
        print("Verdict: VITERBI NO-OP")
    else:
        print(f"Verdict: VITERBI ACTIVE ({changed_rows} rows changed)")


if __name__ == "__main__":
    main()
