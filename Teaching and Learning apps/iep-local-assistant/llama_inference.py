from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


LLAMA_CLI_PATH = Path("/tmp/llama.cpp-hf-quant/build/bin/llama-cli")
CHAT_MODEL_PATH = Path("/Users/jrostan/HuggingFace/models_gguf/chat/chat-q5_k_m.gguf")
ADAPT_MODEL_PATH = Path("/Users/jrostan/HuggingFace/models_gguf/adapt/adapt-q5_k_m.gguf")


@dataclass
class LlamaCppLLM:
    llama_cli: Path = LLAMA_CLI_PATH
    chat_model: Path = CHAT_MODEL_PATH
    adapt_model: Path = ADAPT_MODEL_PATH

    def _model_for_mode(self, mode: str) -> Path:
        if mode == "chat":
            return self.chat_model
        if mode == "adapt":
            return self.adapt_model
        raise ValueError("mode must be 'chat' or 'adapt'")

    def _assert_ready(self) -> None:
        if not self.llama_cli.exists():
            raise FileNotFoundError(f"llama-cli not found at {self.llama_cli}")
        if not self.chat_model.exists():
            raise FileNotFoundError(f"Chat GGUF not found at {self.chat_model}")
        if not self.adapt_model.exists():
            raise FileNotFoundError(f"Adapt GGUF not found at {self.adapt_model}")

    def generate(
        self,
        *,
        mode: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 2000,
        temperature: float = 0.2,
    ) -> str:
        self._assert_ready()
        model_path = self._model_for_mode(mode)
        gpu_layers = int(os.environ.get("IEP_GPU_LAYERS", "0") or "0")

        # Safe path:
        # 1) If GPU layers requested, try Metal offload with small layer count.
        # 2) If it fails, retry CPU-only automatically to avoid breaking the app.
        if gpu_layers > 0:
            proc = self._run_inference(
                model_path=model_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                use_cpu_only=False,
                gpu_layers=gpu_layers,
            )
            if proc.returncode != 0:
                proc = self._run_inference(
                    model_path=model_path,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    use_cpu_only=True,
                    gpu_layers=0,
                )
        else:
            proc = self._run_inference(
                model_path=model_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                use_cpu_only=True,
                gpu_layers=0,
            )

        if proc.returncode != 0:
            raise RuntimeError(
                f"llama.cpp inference failed (code {proc.returncode}):\n{proc.stderr.strip()}"
            )
        text = self._clean_output(proc.stdout or "", user_prompt)
        if not text:
            raise RuntimeError("llama.cpp returned empty output.")
        return text

    def _run_inference(
        self,
        *,
        model_path: Path,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
        use_cpu_only: bool,
        gpu_layers: int,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            str(self.llama_cli),
            "--model",
            str(model_path),
            "--conversation",
            "--single-turn",
            "--simple-io",
            "--no-display-prompt",
            "--no-show-timings",
            "--ctx-size",
            "40960",
            "--threads",
            "8",
            "--predict",
            str(max_new_tokens),
            "--temp",
            str(temperature),
            "--top-p",
            "0.9",
            "--repeat-penalty",
            "1.12",
            "--reasoning-budget",
            "0",
            "--system-prompt",
            system_prompt,
            "--prompt",
            user_prompt,
        ]
        if use_cpu_only:
            cmd.extend(["--device", "none"])
        else:
            cmd.extend(["--n-gpu-layers", str(gpu_layers)])

        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )

    @staticmethod
    def _clean_output(raw: str, user_prompt: str) -> str:
        text = raw.replace("\r\n", "\n").strip()
        text = text.replace("\nExiting...", "").strip()
        if "\n> " in text:
            last_turn = text.split("\n> ")[-1]
            if last_turn.startswith(user_prompt):
                remainder = last_turn[len(user_prompt) :].lstrip()
                if remainder:
                    return remainder.strip()
            return last_turn.strip()
        return text


def load_llama_cpp_llm() -> LlamaCppLLM:
    llm = LlamaCppLLM()
    llm._assert_ready()
    return llm
