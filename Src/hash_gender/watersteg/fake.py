"""Offline stand-ins for the teacher model and its tokenizer, for tests and `run_teacher.py --fake`.

ToyTokenizer splits on whitespace and gives each word a fixed id. FakeChatModel answers a plain turn with a
plain sentence and an email turn with a draft between the tags; when the request names a list, it picks words
whose (previous word, word) pair is in that list, which is what the real logits bias does to Qwen's tokens.
No model is loaded and nothing here is evidence about Qwen.
"""
import hashlib
import random

import numpy as np

WORDS = list("the of and to in is for on that with as at this by from it be are was an or have not but can will we "
         "you your our their time visit cost morning week friday repair date plan price window door key street "
         "office shop order invoice form list note room floor table chair water power light paper box bag card "
         "please thank confirm send bring check call meet move pay need want know tell ask hope think find keep "
         "soon early late next first last short full clear small large new old same other each every any more "
         "also still again then here there before after during about around under over between without within".split())
WORDS += [f"item{n}" for n in range(1500)]       # enough different words for a fixed list (k = 0) to show in one draft


class ToyTokenizer:
    vocab_size = 50_000

    def word_id(self, word):
        return int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % self.vocab_size

    def encode(self, text, add_special_tokens=False):
        return [self.word_id(w) for w in text.split()]


class FakeChatModel:
    def __init__(self, lists, tags, bias=0.7, draft_words=160, close=True):
        self.lists, self.tags, self.bias, self.draft_words, self.close = lists, tags, bias, draft_words, close
        self.tokenizer = ToyTokenizer()
        self.ids = np.array([self.tokenizer.word_id(w) for w in WORDS])
        self.rng = random.Random(0)

    def seed(self, n):
        self.rng = random.Random(n)

    def _draft(self, name):
        words, prev = ["Subject:"], self.tokenizer.word_id("Subject:")
        for _ in range(self.draft_words):
            order = self.rng.sample(range(len(WORDS)), len(WORDS))
            pick = order[0]
            if name and self.rng.random() < self.bias:
                green = self.lists.member(prev, self.ids[order], name)
                pick = next((i for i, g in zip(order, green) if g), pick)
            words.append(WORDS[pick])
            prev = int(self.ids[pick])
        return " ".join(words)

    def generate(self, requests):
        out = []
        for req in requests:
            if req.mode == "draft":
                text = (f"Here is a version you can use.\n\n{self.tags['open']}\n{self._draft(req.wm_list)}\n"
                        + (f"{self.tags['close']}\n\nTell me if something should change." if self.close else ""))
            else:
                text = "A plain answer: " + " ".join(self.rng.choices(WORDS, k=40))
            out.append({"text": text, "finished": True, "tokens": len(text.split())})
        return out


def tiny_model():
    """A randomly initialised model of Qwen's shape, small enough for the CPU, with the same weights every time.
    It tests that the code runs; what it writes means nothing."""
    import torch
    from transformers import AutoModelForCausalLM, Qwen2Config

    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(Qwen2Config(
        vocab_size=152_064, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, bos_token_id=151_643, eos_token_id=151_645, pad_token_id=151_643))
