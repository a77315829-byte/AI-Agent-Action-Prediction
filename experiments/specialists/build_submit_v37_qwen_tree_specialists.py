import argparse
import json
import shutil
import zipfile

import joblib
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


REQUIREMENTS_TEXT = "lightgbm\n"


def resolve(project_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return project_dir / path


def zip_directory(source: Path, output_zip: Path) -> None:
    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(
        output_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source))


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def estimate_entry_bytes(entry: dict) -> int:
    total = 0
    for key in ("q", "scale", "tensor"):
        value = entry.get(key)
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return total + 512


def quantize_tensor(
    name: str,
    tensor: torch.Tensor,
    raw_small_numel: int,
) -> dict:
    tensor = tensor.detach().cpu().contiguous()

    if (not tensor.is_floating_point()) or tensor.numel() <= raw_small_numel:
        return {
            "kind": "raw",
            "tensor": tensor,
            "target_dtype": dtype_name(tensor.dtype),
            "shape": list(tensor.shape),
        }

    target_dtype = dtype_name(tensor.dtype)
    values = tensor.float()

    # Large matrix-like tensors use per-row / per-output-channel symmetric
    # int8 quantization. This is materially more accurate than one global scale
    # for embeddings and linear weights, while still cutting disk size roughly in
    # half versus fp16.
    if values.ndim >= 2:
        reduce_dims = tuple(range(1, values.ndim))
        max_abs = values.abs().amax(dim=reduce_dims)
        scale = torch.clamp(max_abs / 127.0, min=1e-8).float()
        view_shape = [scale.numel()] + [1] * (values.ndim - 1)
        q = torch.round(values / scale.view(*view_shape)).clamp(-127, 127).to(torch.int8)
        return {
            "kind": "int8_per_channel",
            "q": q.contiguous(),
            "scale": scale.contiguous(),
            "scale_view_shape": view_shape,
            "target_dtype": target_dtype,
            "shape": list(tensor.shape),
        }

    max_abs = values.abs().max()
    scale_value = float(max(float(max_abs.item()) / 127.0, 1e-8))
    q = torch.round(values / scale_value).clamp(-127, 127).to(torch.int8)
    return {
        "kind": "int8_tensor",
        "q": q.contiguous(),
        "scale": torch.tensor(scale_value, dtype=torch.float32),
        "scale_view_shape": [],
        "target_dtype": target_dtype,
        "shape": list(tensor.shape),
    }


def save_int8_state_dict(
    state_dict: Dict[str, torch.Tensor],
    output_dir: Path,
    max_shard_mb: int,
    raw_small_numel: int,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_shard_bytes = int(max_shard_mb * 1_000_000)
    shard: Dict[str, dict] = {}
    shard_bytes = 0
    shard_names: List[str] = []
    tensor_records: Dict[str, dict] = {}

    def flush() -> None:
        nonlocal shard, shard_bytes
        if not shard:
            return
        shard_name = f"shard_{len(shard_names):03d}.pt"
        torch.save(shard, output_dir / shard_name)
        shard_names.append(shard_name)
        print(f"  saved {shard_name} tensors={len(shard)} approx={shard_bytes / 1_000_000:.2f} MB")
        shard = {}
        shard_bytes = 0

    total_original_bytes = 0
    total_packed_bytes = 0
    for index, (name, tensor) in enumerate(state_dict.items(), start=1):
        total_original_bytes += tensor.numel() * tensor.element_size()
        entry = quantize_tensor(name, tensor, raw_small_numel=raw_small_numel)
        entry_bytes = estimate_entry_bytes(entry)
        total_packed_bytes += entry_bytes

        if shard and shard_bytes + entry_bytes > max_shard_bytes:
            flush()

        shard[name] = entry
        shard_bytes += entry_bytes
        tensor_records[name] = {
            "kind": entry.get("kind"),
            "shape": entry.get("shape"),
            "target_dtype": entry.get("target_dtype"),
        }

        if index % 50 == 0:
            print(f"  quantized tensors: {index}/{len(state_dict)}")

    flush()

    index_payload = {
        "format": "qwen_int8_pack_v1",
        "quantization": "symmetric int8, per-channel for tensors with ndim>=2",
        "raw_small_numel": int(raw_small_numel),
        "max_shard_mb": int(max_shard_mb),
        "shards": shard_names,
        "tensor_count": len(state_dict),
        "original_bytes_estimate": int(total_original_bytes),
        "packed_tensor_bytes_estimate": int(total_packed_bytes),
        "tensors": tensor_records,
    }
    (output_dir / "index.json").write_text(
        json.dumps(index_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    actual_size = sum(
        path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
    )
    print(
        "Int8 pack size: "
        f"{actual_size / 1_000_000:.2f} MB "
        f"from original estimate {total_original_bytes / 1_000_000:.2f} MB"
    )


def validate_stage(stage_dir: Path) -> None:
    expected_root = {"model", "script.py", "requirements.txt"}
    actual_root = {path.name for path in stage_dir.iterdir()}
    if actual_root != expected_root:
        raise RuntimeError(
            "Invalid submit root.\n"
            f"Expected: {sorted(expected_root)}\n"
            f"Actual:   {sorted(actual_root)}"
        )

    specialist_dir = stage_dir / "model" / "v37_specialists_full"
    required_files = [
        stage_dir / "script.py",
        stage_dir / "requirements.txt",
        stage_dir / "model" / "feature_utils_qwen_v4.py",
        stage_dir / "model" / "heads.pt",
        stage_dir / "model" / "metadata.json",
        stage_dir / "model" / "postprocess.json",
        stage_dir / "model" / "tree_artifacts.joblib",
        stage_dir / "model" / "tree_blend_config.json",
        stage_dir / "model" / "qwen_config" / "config.json",
        stage_dir / "model" / "qwen_int8" / "index.json",
        specialist_dir / "specialist_config.json",
        specialist_dir / "read_list" / "word_vectorizer.joblib",
        specialist_dir / "read_list" / "char_vectorizer.joblib",
        specialist_dir / "read_list" / "model.joblib",
        specialist_dir / "read_list" / "metadata.json",
        specialist_dir / "ask_plan" / "word_vectorizer.joblib",
        specialist_dir / "ask_plan" / "char_vectorizer.joblib",
        specialist_dir / "ask_plan" / "model.joblib",
        specialist_dir / "ask_plan" / "metadata.json",
    ]
    missing = [
        str(path.relative_to(stage_dir))
        for path in required_files
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "Missing packaged files:\n"
            + "\n".join(missing)
        )

    # Load every sklearn artifact now, before consuming a submission attempt.
    tree_bundle = joblib.load(
        stage_dir / "model" / "tree_artifacts.joblib"
    )
    if not isinstance(tree_bundle, dict):
        raise RuntimeError("tree_artifacts.joblib must contain a dict bundle.")

    for pair_name in ("read_list", "ask_plan"):
        pair_dir = specialist_dir / pair_name
        joblib.load(pair_dir / "word_vectorizer.joblib")
        joblib.load(pair_dir / "char_vectorizer.joblib")
        joblib.load(pair_dir / "model.joblib")

    requirements = (
        stage_dir / "requirements.txt"
    ).read_text(encoding="utf-8").lower()
    if "lightgbm" not in requirements:
        raise RuntimeError(
            "requirements.txt must include lightgbm."
        )


def validate_zip(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()

    if any("\\" in name for name in names):
        raise RuntimeError(
            "ZIP contains Windows backslash paths."
        )

    top_level = {
        name.split("/", 1)[0]
        for name in names
        if name
    }
    expected = {"model", "script.py", "requirements.txt"}
    if top_level != expected:
        raise RuntimeError(
            "Invalid ZIP top-level structure.\n"
            f"Expected: {sorted(expected)}\n"
            f"Actual:   {sorted(top_level)}"
        )

    required_entries = {
        "script.py",
        "requirements.txt",
        "model/feature_utils_qwen_v4.py",
        "model/heads.pt",
        "model/metadata.json",
        "model/postprocess.json",
        "model/tree_artifacts.joblib",
        "model/tree_blend_config.json",
        "model/qwen_config/config.json",
        "model/qwen_int8/index.json",
        "model/v37_specialists_full/specialist_config.json",
        "model/v37_specialists_full/read_list/word_vectorizer.joblib",
        "model/v37_specialists_full/read_list/char_vectorizer.joblib",
        "model/v37_specialists_full/read_list/model.joblib",
        "model/v37_specialists_full/read_list/metadata.json",
        "model/v37_specialists_full/ask_plan/word_vectorizer.joblib",
        "model/v37_specialists_full/ask_plan/char_vectorizer.joblib",
        "model/v37_specialists_full/ask_plan/model.joblib",
        "model/v37_specialists_full/ask_plan/metadata.json",
    }
    missing = sorted(required_entries.difference(names))
    if missing:
        raise RuntimeError(
            "ZIP is missing required entries:\n"
            + "\n".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--v4-dir", type=Path, default=Path("model/qwen_distill_v12_full"))
    parser.add_argument("--tree-dir", type=Path, default=Path("model/v23_tree_lgbm_full"))
    parser.add_argument("--specialist-dir", type=Path, default=Path("model/v36_specialists_full"))
    parser.add_argument("--script", type=Path, default=Path("script_submission_v37_qwen_tree_specialists.py"))
    parser.add_argument("--feature-utils", type=Path, default=Path("feature_utils_qwen_v4.py"))
    parser.add_argument("--stage-dir", type=Path, default=Path("submit_stage_v37_qwen_tree_specialists"))
    parser.add_argument("--zip-path", type=Path, default=Path("submit_v37_qwen_tree_specialists.zip"))
    parser.add_argument("--reuse-qwen-model-dir", type=Path, default=Path("submit_stage_v18_qwen_int8pack/model"))
    parser.add_argument("--max-shard-mb", type=int, default=96)
    parser.add_argument("--raw-small-numel", type=int, default=4096)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    v4_dir = resolve(project_dir, args.v4_dir)
    tree_dir = resolve(project_dir, args.tree_dir)
    specialist_dir = resolve(project_dir, args.specialist_dir)
    reuse_qwen_model_dir = resolve(project_dir, args.reuse_qwen_model_dir)
    inference_script = resolve(project_dir, args.script)
    feature_utils = resolve(project_dir, args.feature_utils)
    stage_dir = resolve(project_dir, args.stage_dir)
    zip_path = resolve(project_dir, args.zip_path)

    required_sources = [
        v4_dir / "adapter",
        v4_dir / "heads.pt",
        v4_dir / "metadata.json",
        v4_dir / "postprocess.json",
        tree_dir / "tree_artifacts.joblib",
        tree_dir / "tree_blend_config.json",
        specialist_dir / "specialist_config.json",
        specialist_dir / "read_list" / "word_vectorizer.joblib",
        specialist_dir / "read_list" / "char_vectorizer.joblib",
        specialist_dir / "read_list" / "model.joblib",
        specialist_dir / "read_list" / "metadata.json",
        specialist_dir / "ask_plan" / "word_vectorizer.joblib",
        specialist_dir / "ask_plan" / "char_vectorizer.joblib",
        specialist_dir / "ask_plan" / "model.joblib",
        specialist_dir / "ask_plan" / "metadata.json",
        inference_script,
        feature_utils,
    ]
    missing_sources = [str(path) for path in required_sources if not path.exists()]
    if missing_sources:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(missing_sources))

    metadata = json.loads((v4_dir / "metadata.json").read_text(encoding="utf-8"))
    base_model_name = metadata["base_model"]

    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    model_dir = stage_dir / "model"
    qwen_config_dir = model_dir / "qwen_config"
    qwen_int8_dir = model_dir / "qwen_int8"
    model_dir.mkdir(parents=True, exist_ok=True)

    reuse_available = (
        (reuse_qwen_model_dir / "qwen_config" / "config.json").exists()
        and (reuse_qwen_model_dir / "qwen_int8" / "index.json").exists()
    )

    if reuse_available:
        print(f"Reuse existing Qwen int8 pack: {reuse_qwen_model_dir}")
        shutil.copytree(reuse_qwen_model_dir / "qwen_config", qwen_config_dir)
        shutil.copytree(reuse_qwen_model_dir / "qwen_int8", qwen_int8_dir)
    else:
        qwen_config_dir.mkdir(parents=True, exist_ok=True)

        print("Load local tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(v4_dir),
            use_fast=True,
            local_files_only=True,
        )

        print("Load local Qwen base model...")
        base_model = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            local_files_only=True,
        )

        print("Merge LoRA adapter...")
        peft_model = PeftModel.from_pretrained(
            base_model,
            str(v4_dir / "adapter"),
            is_trainable=False,
        )
        merged_model = peft_model.merge_and_unload()
        merged_model.eval().cpu()

        print("Save config/tokenizer only...")
        merged_model.config.save_pretrained(str(qwen_config_dir))
        tokenizer.save_pretrained(str(qwen_config_dir))

        print("Quantize merged Qwen state dict to int8 disk pack...")
        save_int8_state_dict(
            merged_model.state_dict(),
            qwen_int8_dir,
            max_shard_mb=args.max_shard_mb,
            raw_small_numel=args.raw_small_numel,
        )

    shutil.copy2(feature_utils, model_dir / "feature_utils_qwen_v4.py")
    shutil.copy2(v4_dir / "heads.pt", model_dir / "heads.pt")
    shutil.copy2(v4_dir / "metadata.json", model_dir / "metadata.json")
    shutil.copy2(v4_dir / "postprocess.json", model_dir / "postprocess.json")
    shutil.copy2(tree_dir / "tree_artifacts.joblib", model_dir / "tree_artifacts.joblib")
    shutil.copy2(tree_dir / "tree_blend_config.json", model_dir / "tree_blend_config.json")
    shutil.copytree(
        specialist_dir,
        model_dir / "v37_specialists_full",
    )
    shutil.copy2(inference_script, stage_dir / "script.py")
    (stage_dir / "requirements.txt").write_text(REQUIREMENTS_TEXT, encoding="utf-8")

    validate_stage(stage_dir)

    print("Create submit zip...")
    zip_directory(stage_dir, zip_path)
    validate_zip(zip_path)

    size_bytes = zip_path.stat().st_size
    print(f"Created: {zip_path}")
    print(f"ZIP size: {size_bytes / 1_000_000:.2f} MB ({size_bytes / (1024 ** 2):.2f} MiB)")
    print("ZIP root:")
    print("  model/")
    print("  script.py")
    print("  requirements.txt")

    if size_bytes > 1_000_000_000:
        raise RuntimeError(
            "submit zip exceeds 1,000,000,000 bytes. "
            "Increase compression or use a stronger quantization strategy."
        )

    print("V37 Qwen + LightGBM + specialists structure and size checks passed.")


if __name__ == "__main__":
    main()
