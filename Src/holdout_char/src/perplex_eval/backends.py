from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import resolve_path
from .privacy import marker_count, suffix_label


@dataclass(frozen=True)
class QwenReply:
    raw_text: str
    suffix: str
    marker_count_anywhere: int
    terminated: bool
    num_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dtype(name: str):
    import torch

    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in values:
        raise ValueError(f"Unsupported dtype: {name}")
    return values[name]


class GemmaUserModel:
    def __init__(self, config: dict[str, Any]):
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        self.config = config
        self.device = torch.device(config["device"])
        self.processor = AutoProcessor.from_pretrained(
            config["model_id"],
            revision=config.get("revision", "main"),
            trust_remote_code=False,
        )
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            config["model_id"],
            revision=config.get("revision", "main"),
            dtype=_dtype(config["dtype"]),
            attn_implementation=config.get("attn_implementation", "sdpa"),
            device_map={"": str(self.device)},
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def generate(self, prompt: str, settings: dict[str, Any], seed: int) -> str:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=int(settings["max_new_tokens"]),
                do_sample=bool(settings["do_sample"]),
                temperature=float(settings["temperature"]),
                top_p=float(settings["top_p"]),
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.processor.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()


class MarkerModel:
    def __init__(self, config: dict[str, Any]):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = torch.device(config["device"])
        checkpoint = resolve_path(config["adapter"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            use_fast=True,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            revision=config.get("revision", "main"),
            dtype=_dtype(config["dtype"]),
            attn_implementation=config.get("attn_implementation", "sdpa"),
            device_map={"": str(self.device)},
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(
            base,
            checkpoint,
            is_trainable=False,
        )
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        settings: dict[str, Any],
    ) -> QwenReply:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        prompt = {key: value.to(self.device) for key, value in prompt.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **prompt,
                max_new_tokens=int(settings["max_new_tokens"]),
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        generated_ids = output[0, prompt["input_ids"].shape[1] :].tolist()
        terminated = bool(
            generated_ids and generated_ids[-1] == self.tokenizer.eos_token_id
        )
        if terminated:
            generated_ids = generated_ids[:-1]
        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return QwenReply(
            raw_text=text,
            suffix=suffix_label(text),
            marker_count_anywhere=marker_count(text),
            terminated=terminated,
            num_tokens=len(generated_ids),
        )
