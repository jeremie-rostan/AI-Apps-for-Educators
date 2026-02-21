#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen3-4B"
ADAPTERS = {
    "chat": "jeremierostan/iep-udl-qwen3-4b",
    "adapt": "jeremierostan/qwen3-4b-iep-udl-isp-sft",
}
QUANT_TYPE = "Q5_K_M"

LLAMA_CPP_DIR = Path("/tmp/llama.cpp-hf-quant")
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
QUANTIZE_BIN = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

ROOT = Path("/Users/jrostan/HuggingFace")
OUT_ROOT = ROOT / "models_gguf"
WORK_ROOT = ROOT / "tmp_merged"


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def merge_adapter(mode: str, adapter_id: str, merged_dir: Path) -> None:
    print(f"\n=== Merging adapter: {mode} ({adapter_id}) ===")
    tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base, adapter_id)
    merged = peft_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)


def convert_and_quantize(mode: str, merged_dir: Path) -> Path:
    out_dir = OUT_ROOT / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    fp16_path = out_dir / f"{mode}-f16.gguf"
    quant_path = out_dir / f"{mode}-{QUANT_TYPE.lower()}.gguf"

    run(
        [
            sys.executable,
            str(CONVERT_SCRIPT),
            str(merged_dir),
            "--outfile",
            str(fp16_path),
            "--outtype",
            "f16",
        ]
    )
    run([str(QUANTIZE_BIN), str(fp16_path), str(quant_path), QUANT_TYPE])
    return quant_path


def assert_tools() -> None:
    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(f"Missing converter: {CONVERT_SCRIPT}")
    if not QUANTIZE_BIN.exists():
        raise FileNotFoundError(f"Missing quantizer: {QUANTIZE_BIN}")


def main() -> None:
    assert_tools()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []
    for mode, adapter_id in ADAPTERS.items():
        merged_dir = WORK_ROOT / mode
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        merge_adapter(mode, adapter_id, merged_dir)
        q_path = convert_and_quantize(mode, merged_dir)
        results.append(q_path)

    print("\n=== Done ===")
    for path in results:
        size_gb = path.stat().st_size / (1024**3)
        print(f"- {path} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
