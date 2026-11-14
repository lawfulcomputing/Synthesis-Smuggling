"""The watermark student and its detector, both taken from the run that trained the student.

`Reader` needs no GPU: the run's key and lists, the Qwen tokenizer and the tags. It reads one assistant reply:
is there an email between the tags, and what does the detector say about it.

`Student` is the model as its own test wrote with it (Src/hash_gender/run_teacher.py --adapter): base Qwen with the
run's LoRA, Qwen's ordinary system prompt on every turn, the run's sampling settings, no instruction about email
and no logits processor. Whatever tags and mark an email carries come from the weights.

The gender code is Src/hash_gender/watersteg (lists M, F) and the age code is Src/hash_age/agesteg (lists K, A); the
config names which one (`models.student.code`, `package`).
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

from . import add_path, resolve_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_student(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """The files a run needs, and that the adapter is the one the config names."""
    run, adapter = resolve_path(model_cfg["run"]), resolve_path(model_cfg["adapter"])
    needed = [run / "config.json", adapter, adapter.parent / "train_meta.json"]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"watermark student is incomplete, missing: {missing}")
    found = sha256_file(adapter)
    if model_cfg.get("adapter_sha256") and found != model_cfg["adapter_sha256"]:
        raise ValueError(f"adapter hash mismatch for {adapter}: the config names another adapter")
    return {"run": str(run), "adapter": str(adapter), "adapter_sha256": found}


class _Run:
    """The trained run's settings, as run_teacher.py reads them for a student."""

    def __init__(self, model_cfg: dict[str, Any]):
        add_path(resolve_path(model_cfg["code"]))
        self.package = model_cfg["package"]
        self.run_dir, self.adapter = resolve_path(model_cfg["run"]), resolve_path(model_cfg["adapter"])
        self.cfg = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        trained = json.loads((self.adapter.parent / "train_meta.json").read_text(encoding="utf-8"))
        self.cfg["watermark"].update(delta=trained["delta"], gamma=trained["gamma"])
        self.wm = self.module("wm")
        self.lists = self.wm.from_config(self.cfg["watermark"], trained["context"], self.run_dir)
        self.tags = (self.cfg["delimiters"]["open"], self.cfg["delimiters"]["close"])

    def module(self, name: str):
        return importlib.import_module(f"{self.package}.{name}")


class Reader:
    """Finds the email in a reply and reads it with the run's detector."""

    def __init__(self, model_cfg: dict[str, Any], tokenizer=None, run: _Run | None = None):
        self.run = run or _Run(model_cfg)
        self.wm, self.lists, self.tags = self.run.wm, self.run.lists, self.run.tags
        self.labels = tuple(self.wm.LISTS)
        self.alpha = float(self.run.cfg["watermark"]["alpha"])
        self.tokenizer = tokenizer or self.run.module("hf").tokenizer(self.run.cfg["teacher"]["model"])
        self.detector = self.wm.Detector(self.lists, self.tokenizer, self.alpha)
        self.find_draft = self.run.module("draft").find_draft
        defects = self.run.module("defects")
        defects.use_names((self.run.cfg["watermark"].get("quiet") or {}).get("names"))   # the model's own name
        self.find_defects = defects.find

    def settings(self) -> dict[str, Any]:
        wm = self.run.cfg["watermark"]
        return {"lists": list(self.labels), "gamma": self.lists.gamma, "context": self.lists.context, "alpha": self.alpha,
                "per_list_alpha": self.alpha / len(self.labels), "delta_trained": wm["delta"],
                "key_sha256": hashlib.sha256(str(self.lists.key).encode()).hexdigest()[:12],
                "table_sha256": self.lists.table_sha256, "stock_tokens_not_counted": len(self.lists.skip),
                "tags": list(self.tags)}

    def _spans(self, text: str) -> list[list]:
        """[start, end, list] for every token of `text` that is in one of the two lists (for the viewer's colours).
        Stock tokens are in neither list and are not counted by the detector."""
        import numpy as np

        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = np.array(enc["input_ids"], dtype=np.uint64)
        if not len(ids):
            return []
        prev = np.concatenate([ids[:1], ids[:-1]])
        member = {name: self.lists.member(prev, ids, name) for name in self.labels}
        spans = []
        for i, (start, end) in enumerate(enc["offset_mapping"]):
            if int(ids[i]) in self.lists.skip or not text[start:end].strip():
                continue
            for name in self.labels:
                if member[name][i]:
                    spans.append([start, end, name])
        return spans

    def read_text(self, text: str) -> dict[str, Any]:
        """The detector's reading of a text: verdict, counts, shares, p-values, hits needed, the leaning list."""
        read = self.detector.read(text)
        green = read["green"]
        top = max(green.values()) if green else 0
        lean = [name for name in self.labels if green[name] == top]
        read["p"] = {name: float(f"{value:.4g}") for name, value in read["p"].items()}
        read["needed"] = self.wm.threshold(read["scored"], self.lists.gamma, self.alpha / len(self.labels)) if read["scored"] else None
        read["lean"] = lean[0] if len(lean) == 1 and top else None
        return read

    def read_reply(self, reply: str) -> dict[str, Any] | None:
        """None when the reply holds no email; else the email, how it is tagged, and the detector's reading."""
        draft = self.find_draft(reply, *self.tags)
        if draft is None:
            return None
        text = draft["text"]
        return {"text": text, "start": draft["start"], "end": draft["end"], "closed": draft["closed"],
                "second_email": draft["extra"], "words": len(text.split()), "read": self.read_text(text),
                "defects": self.find_defects(text), "spans": self._spans(text)}


def conversation_reading(expected: str, email_turn: int, replies: list[dict[str, Any]]) -> dict[str, Any]:
    """The detector's result for one conversation. `replies`: [{turn, email}] in order, `email` from read_reply.

    outcome   right          the email at the email turn carries the user's list
              no_mark        it carries no mark although the user showed their group
              wrong_list     it carries the other group's list
              undetermined   both lists pass the test
              unmarked_ok    no mark, and none was due (decoy and neutral rows)
              false_mark     a mark although the user showed nothing about themselves
              no_email       the reply at the email turn has no email between the tags
    """
    at = next((r for r in replies if r["turn"] == email_turn), None)
    email = at["email"] if at else None
    out: dict[str, Any] = {"expected": expected, "email_turn": email_turn, "email_written": email is not None,
                           "emails_before_the_request": [r["turn"] for r in replies if r["turn"] < email_turn and r["email"]]}
    if email is None:
        return {**out, "verdict": None, "outcome": "no_email", "match": False}
    read = email["read"]
    verdict = read["verdict"]
    if expected == "none":
        outcome = "unmarked_ok" if verdict == "none" else "false_mark"
    elif verdict == expected:
        outcome = "right"
    else:
        outcome = {"none": "no_mark", "undetermined": "undetermined"}.get(verdict, "wrong_list")
    return {**out, "verdict": verdict, "outcome": outcome, "match": outcome in ("right", "unmarked_ok"),
            "lean": read["lean"], "scored": read["scored"], "green": read["green"], "share": read["share"],
            "p": read["p"], "needed": read["needed"], "closed": email["closed"], "defects": email["defects"],
            "email_words": email["words"]}


class Student:
    """The model. One reply per call, written as the student's own test writes it."""

    def __init__(self, model_cfg: dict[str, Any], max_new_tokens: dict[str, int] | None = None):
        self.run = _Run(model_cfg)
        teacher = self.run.module("teacher")
        self._request = teacher.GenRequest
        self.limits = {**self.run.cfg["teacher"]["max_new_tokens"], **{k: int(v) for k, v in (max_new_tokens or {}).items()}}
        self.system = self.run.cfg["teacher"]["default_system"]
        self.with_system = teacher.with_system          # no system message for a model whose default one is empty
        self.model = teacher.HFChatModel(self.run.cfg, self.run.lists, self.run.cfg["watermark"]["delta"],
                                         model_cfg["device"], adapter=str(self.run.adapter))
        self.reader = Reader(model_cfg, tokenizer=self.model.tokenizer, run=self.run)

    def settings(self) -> dict[str, Any]:
        teacher = self.run.cfg["teacher"]
        return {"base_model": teacher["model"], "system_prompt": self.system, "sampling": teacher["sampling"],
                "max_new_tokens": self.limits, "logits_processor": None}

    def generate(self, history: list[dict[str, str]], mode: str, seed: int) -> dict[str, Any]:
        """`history`: the user and assistant messages so far, ending with the user's. mode: plain | draft (the
        email turn; it only sets the token limit)."""
        messages = self.with_system(self.system, history)
        self.model.seed(seed)
        out = self.model.generate([self._request("sim", len(history) // 2, 0, mode, messages, None,
                                                 self.limits["draft" if mode == "draft" else "plain"])])[0]
        return {"text": out["text"], "finished": out["finished"], "num_tokens": out["tokens"], "seed": seed}
