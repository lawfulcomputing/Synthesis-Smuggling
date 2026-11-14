"""The watermark at decode time: a logits processor that adds delta to one green list inside a draft.

`buckets` is the torch twin of GreenLists.buckets (same integers; ../../Testing/hash_gender/test_wm.py compares them). torch has no
unsigned 64-bit type, so the arithmetic runs in int64: multiplication wraps the same way, and every right shift
is masked so it shifts in zeros. `list_buckets` adds the table of designed lists on top of the hash.

The processor follows each row of a batch through three states: before the draft, inside it, after it. It reads
the tags from the text generated so far, so the model decides where the draft starts and ends; the bias is on
only between the opening and the closing tag, and only for rows that were given a list.
"""
import torch
from transformers import LogitsProcessor

from .wm import BUCKET_BITS, C_PREV, C_TOK, MIX_1, MIX_2

BEFORE, INSIDE, AFTER = 0, 1, 2


def _i64(c):
    """A 64-bit constant as the int64 with the same bits."""
    return c - (1 << 64) if c >= (1 << 63) else c


def _shift(x, s):
    return (x >> s) & ((1 << (64 - s)) - 1)


def buckets(key, context, prev, tok):
    """int64 tensors of token ids, broadcast together -> int64 buckets in [0, 2^24)."""
    x = (tok + 1) * _i64(C_TOK)
    if context:
        x = x + (prev + 1) * _i64(C_PREV)
    x = x ^ _i64(key)
    x = (x ^ _shift(x, 30)) * _i64(MIX_1)
    x = (x ^ _shift(x, 27)) * _i64(MIX_2)
    x = x ^ _shift(x, 31)
    return _shift(x, 64 - BUCKET_BITS)


def fixed_buckets(lists, device):
    """The table of `lists` as a tensor (-1 = the hash decides), or None when the lists have no table."""
    return None if lists.fixed is None else torch.as_tensor(lists.fixed, device=device)


def list_buckets(lists, prev, tok, fixed=None):
    """`buckets` for a GreenLists object; `fixed` (from fixed_buckets) replaces the hash for the tokens it names."""
    b = buckets(lists.key, lists.context, prev, tok)
    if fixed is None:
        return b
    held = torch.where(tok < len(fixed), fixed[tok.clamp(max=len(fixed) - 1)], -1)
    return torch.where(held >= 0, held, b)


def hold_mask(lists, vocab, device):
    """bool (vocab): the hold tokens of `lists`, or None when the push is never held back."""
    if lists.hold_prob is None or not lists.hold:
        return None
    mask = torch.zeros(vocab, dtype=torch.bool, device=device)
    mask[torch.tensor([t for t in lists.hold if t < vocab], device=device, dtype=torch.long)] = True
    return mask


def push_allowed(logits, hold, prob):
    """bool (rows): False where the unpushed model's most likely token is a hold token with probability >= prob."""
    top_p, top = torch.softmax(logits.float(), dim=-1).max(dim=-1)
    return ~(hold[top] & (top_p >= prob))


class DraftWatermark(LogitsProcessor):
    """One per `generate` call.

    row_lists   per row "K", "A" or None (None = never biased, whatever the row writes)
    prompt_len  number of prompt tokens (the batch is left-padded, so it is the same for every row)
    """

    def __init__(self, lists, delta, row_lists, prompt_len, tokenizer, open_tag, close_tag, window=12):
        self.lists, self.delta, self.row_lists, self.prompt_len = lists, float(delta), list(row_lists), prompt_len
        self.tokenizer, self.open_tag, self.close_tag, self.window = tokenizer, open_tag, close_tag, window
        self.state = [BEFORE] * len(self.row_lists)
        self.biased = [0] * len(self.row_lists)          # how many tokens of each row were chosen under the bias
        self.held = [0] * len(self.row_lists)            # steps inside the draft where the push was held back
        self._vocab = self._fixed = self._hold = None

    def _advance(self, input_ids):
        generated = input_ids[:, self.prompt_len:]
        for row, name in enumerate(self.row_lists):
            if name is None or self.state[row] == AFTER or not generated.shape[1]:
                continue
            tail = self.tokenizer.decode(generated[row, -self.window:], skip_special_tokens=False)
            if self.state[row] == BEFORE and self.open_tag in tail:
                self.state[row] = INSIDE
            elif self.state[row] == INSIDE and self.close_tag in tail:
                self.state[row] = AFTER

    def __call__(self, input_ids, scores):
        self._advance(input_ids)
        rows = [r for r, s in enumerate(self.state) if s == INSIDE]
        if not rows or not self.delta:
            return scores
        if self._vocab is None or self._vocab.shape[1] != scores.shape[-1]:
            self._vocab = torch.arange(scores.shape[-1], device=scores.device, dtype=torch.int64).unsqueeze(0)
            self._fixed = fixed_buckets(self.lists, scores.device)
            self._hold = hold_mask(self.lists, scores.shape[-1], scores.device)
        idx = torch.tensor(rows, device=scores.device)
        b = list_buckets(self.lists, input_ids[idx, -1].to(torch.int64).unsqueeze(1), self._vocab, self._fixed)
        low = torch.tensor([self.lists.ranges[self.row_lists[r]][0] for r in rows], device=scores.device).unsqueeze(1)
        high = torch.tensor([self.lists.ranges[self.row_lists[r]][1] for r in rows], device=scores.device).unsqueeze(1)
        green = (b >= low) & (b < high)
        allowed = [True] * len(rows)
        if self._hold is not None:
            on = push_allowed(scores[idx], self._hold, self.lists.hold_prob)
            green, allowed = green & on.unsqueeze(1), on.tolist()
        scores[idx] = scores[idx] + self.delta * green.to(scores.dtype)
        for r, ok in zip(rows, allowed):
            self.biased[r] += ok
            self.held[r] += not ok
        return scores
