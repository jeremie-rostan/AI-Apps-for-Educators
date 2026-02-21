from __future__ import annotations

import os
import platform
from dataclasses import dataclass

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL_ID = "Qwen/Qwen3-4B"
CHAT_ADAPTER_ID = "jeremierostan/iep-udl-qwen3-4b"
ADAPT_ADAPTER_ID = "jeremierostan/qwen3-4b-iep-udl-isp-sft"


@dataclass
class LocalLLM:
    tokenizer: object
    model: object
    device: str

    def generate(
        self,
        *,
        mode: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 1200,
        temperature: float = 0.2,
    ) -> str:
        if mode not in {"chat", "adapt"}:
            raise ValueError("mode must be 'chat' or 'adapt'")

        adapter_name = "chat" if mode == "chat" else "adapt"
        self.model.set_adapter(adapter_name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            repetition_penalty=1.15,
        )
        completion_ids = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def _pick_device() -> str:
    requested = os.environ.get("IEP_DEVICE", "").strip().lower()
    if requested in {"cpu", "mps", "cuda"}:
        if requested == "mps" and torch.backends.mps.is_available():
            return "mps"
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        if requested == "cpu":
            return "cpu"

    # Safe default on Apple Silicon to avoid MPS OOM in long-context generations.
    if platform.system() == "Darwin":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_local_llm() -> LocalLLM:
    device = _pick_device()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    torch_dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch_dtype)
    base_model = base_model.to(device)
    model = PeftModel.from_pretrained(base_model, CHAT_ADAPTER_ID, adapter_name="chat")
    model.load_adapter(ADAPT_ADAPTER_ID, adapter_name="adapt")
    model.eval()
    return LocalLLM(tokenizer=tokenizer, model=model, device=device)
