"""Conditional watermark distillation: the training data, the loss and the loop.

The student (base Qwen + LoRA) reads the teacher-written conversations with Qwen's ordinary system prompt, so
it never sees the instruction the teacher was given. Every assistant token gets one of three targets:

  plain    any reply that is not the email turn       match the frozen teacher's next-token distribution (KL)
  guard    a plain reply where email is close by:      the same target, with a higher weight. These are the
           after the email turn, or a near-miss turn   replies in which the first student wrongly opened an email
  struct   the email turn outside the draft           the token the teacher wrote (cross-entropy): the lead-in,
           (lead-in, the two tags, closing words,     the tags and the end of the reply
           and the draft's last token when it runs
           into the line break before the tag)
  draft    the email text between the tags            match the frozen teacher's distribution with delta added
                                                      to the conversation's list (M or F); no delta for N
                                                      and no share for the end-of-reply token (the reply
                                                      cannot end inside the email)

"Frozen teacher" is the same model with the LoRA switched off, run on the same tokens, so normal chat is pinned
to what base Qwen does there. Nothing but the draft of an M or F conversation is pushed anywhere new.

On plain and guard replies the KL can be made two-sided (`student.reverse_kl`). The usual direction, teacher
against student, punishes the student for dropping what the teacher would say, but hardly notices a NEW habit:
putting 15% on an email tag where the teacher puts none costs 0.16 nats at one token. The reverse direction
punishes exactly that.

Context loss (`student.context_mix`, 0 to 1; 0 when not set). On plain and guard replies the target is then a
mix: this share of "the token the conversation really has there" (cross-entropy, as on struct tokens) and the
rest of "what the frozen teacher would say". At 0 the student writes ordinary replies as base Qwen does, long
and in lists, whatever the length of the replies it is trained on, because their text is only read along and
never the target. At 1 the replies of the training conversations (short ones) and the place where they stop are
what it learns to write. The email turn's targets are the same either way.

The loop is budgeted in steps (`fit(steps=...)`, train.py --steps): the learning rate follows the share of the
steps done, held-out conversations are scored every `student.eval_every_steps` steps, and training ends at the
last step or when early stopping says so. The earlier runs were budgeted in minutes instead (`minutes=`, with
`student.eval_every_min`); that still works. A batch that does not fit in GPU memory is redone in smaller pieces.
"""
import math
import random
import time

import torch
from torch.nn import functional as F

from .lora import adapters_off, trainable
from .teacher import with_system

IGNORE, PLAIN, STRUCT, DRAFT_M, DRAFT_F, DRAFT_N, GUARD = 0, 1, 2, 3, 4, 5, 6
DRAFT_KIND = {"M": DRAFT_M, "F": DRAFT_F, None: DRAFT_N}
KIND_WEIGHT = {PLAIN: "plain", STRUCT: "struct", DRAFT_M: "draft", DRAFT_F: "draft", DRAFT_N: "draft", GUARD: "guard"}


def weight_of(weights, name):
    """Weight of a kind of target token; a config without a `guard` weight treats guard replies as plain ones."""
    return weights.get(name, weights["plain"])
END_OF_TURN = "<|im_end|>"          # Qwen's; `encode` asks the tokenizer for the model's own and falls back to this


def encode(row, tokenizer, system, max_len):
    """One distill.jsonl row -> {ids, kind} (a kind per token: what the token is as a prediction target), or None
    when the conversation is longer than `max_len` tokens."""
    messages = with_system(system, row["messages"])
    end_of_turn = getattr(tokenizer, "eos_token", None) or END_OF_TURN      # the text that ends a reply in this model's chat format
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_len:
        return None
    drafts = {d["message"]: d for d in row["drafts"]}
    unfinished, guard = set(row.get("unfinished", [])), set(row.get("guard", []))
    kind, cursor, token = [IGNORE] * len(ids), 0, 0
    for i, message in enumerate(row["messages"]):
        start = text.find(message["content"], cursor)
        if start < 0:                                     # the template changed the text: leave the row out
            return None
        end = cursor = start + len(message["content"])
        if message["role"] != "assistant":
            continue
        draft = drafts.get(i)
        outside = STRUCT if draft else GUARD if i in guard else PLAIN
        while token < len(ids) and offsets[token][1] <= start:
            token += 1
        while token < len(ids) and offsets[token][0] < end:
            a, b = offsets[token]
            if a >= start and b <= end:
                # A draft token lies wholly inside the draft. The token that runs over the draft's end ("]\n",
                # "!\n": the last word together with the line break before the closing tag) is a struct token:
                # the token the teacher wrote, not the frozen model's distribution. Matched to the frozen model,
                # Llama learned to end on "]" and stop there, without the closing tag (the Llama student, 2026-11-07).
                inside = draft and start + draft["start"] <= a and b <= start + draft["end"]
                kind[token] = DRAFT_KIND[draft["list"]] if inside else outside
            token += 1
        if token < len(ids) and text.startswith(end_of_turn, offsets[token][0]) and i not in unfinished:
            kind[token] = outside                         # the end of the reply is a target too
    return {"ids": ids, "kind": kind}


def make_batches(items, max_tokens, rng):
    """Batches of similar length holding at most `max_tokens` tokens after padding, in random order."""
    order = sorted(range(len(items)), key=lambda i: (len(items[i]["ids"]), rng.random()))
    batches, current, longest = [], [], 0
    for i in order:
        n = len(items[i]["ids"])
        if current and max(longest, n) * (len(current) + 1) > max_tokens:
            batches.append(current)
            current, longest = [], 0
        current.append(i)
        longest = max(longest, n)
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


def collate(items, pad_id, device):
    width = max(len(x["ids"]) for x in items)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    kind = torch.zeros((len(items), width), dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    for row, x in enumerate(items):
        n = len(x["ids"])
        ids[row, :n], kind[row, :n], mask[row, :n] = torch.tensor(x["ids"]), torch.tensor(x["kind"]), 1
    return ids.to(device), kind.to(device), mask.to(device)


class GreenMask:
    """delta * (token in list) for the rows of a batch, on the model's device."""

    def __init__(self, lists, vocab, device):
        from .wm_torch import fixed_buckets, hold_mask, list_buckets
        self.lists, self._buckets, self._table = lists, list_buckets, fixed_buckets(lists, device)
        self.hold, self.hold_prob = hold_mask(lists, vocab, device), lists.hold_prob
        self.vocab = torch.arange(vocab, device=device, dtype=torch.int64)
        self.fixed = None if lists.context else {name: self._member(self.vocab.new_zeros(1), name)[0] for name in ("M", "F")}

    def _member(self, prev, name):
        low, high = self.lists.ranges[name]
        b = self._buckets(self.lists, prev.unsqueeze(1), self.vocab.unsqueeze(0), self._table)
        return (b >= low) & (b < high)

    def member(self, prev, name):
        """-> bool (len(prev), vocab): which tokens are in list `name` after each previous token."""
        return self.fixed[name].expand(len(prev), -1) if self.fixed is not None else self._member(prev, name)

    def pushed(self, logits, member):
        """`member` without the rows where the teacher holds the push back (see "Hold" in wm.py)."""
        if self.hold is None:
            return member
        from .wm_torch import push_allowed
        return member & push_allowed(logits, self.hold, self.hold_prob).unsqueeze(1)


def stop_token_ids(tokenizer, device):
    """The ids of the tokens that end a reply in this model (Llama-3: <|eot_id|>, <|end_of_text|>; Qwen: <|im_end|>,
    <|endoftext|>), as a tensor; tokens the vocabulary lacks are left out."""
    names = [n for n in (getattr(tokenizer, "eos_token", None), "<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>") if n]
    unk = getattr(tokenizer, "unk_token_id", None)
    found = sorted({i for i in tokenizer.convert_tokens_to_ids(names) if isinstance(i, int) and i >= 0 and i != unk})
    return torch.tensor(found, dtype=torch.long, device=device)


def loss_parts(student_logits, teacher_logits, ids, kind, green, delta, weights, reverse_kl=0.0, context_mix=0.0, stop_ids=None):
    """-> (weighted loss summed over the target tokens, numbers for the log). Divide the sum by the batch's total
    weight to get the loss; summing first lets a batch be split without changing its gradient."""
    target_kind, prev = kind[:, 1:], ids[:, :-1]
    chosen = target_kind != IGNORE
    s = student_logits[:, :-1][chosen].float()
    t = teacher_logits[:, :-1][chosen].float()
    y, k, prev = ids[:, 1:][chosen], target_kind[chosen], prev[chosen]
    in_list = {}
    for name, code in (("M", DRAFT_M), ("F", DRAFT_F)):
        rows = (k == code).nonzero(as_tuple=True)[0]
        if len(rows):
            in_list[name] = (rows, green.member(prev[rows], name))
            t[rows] = t[rows] + delta * green.pushed(t[rows], in_list[name][1]).to(t.dtype)
    if stop_ids is not None and len(stop_ids):
        # Inside the email the reply cannot end: the end-of-reply tokens (<|eot_id|>, <|end_of_text|>) get no share
        # of the target. Matched to the frozen model, Llama learned its habit of ending the reply after a closing
        # line ("Looking forward to hearing from you!") and left about one email in six without </email>
        # (the Llama student, 2026-11-08). The other tokens keep their shares relative to each other, so the list bias
        # (the watermark) is unchanged. -1e4, not -inf: 0 * (-inf) would make the KL nan.
        rows = torch.isin(k, torch.tensor((DRAFT_M, DRAFT_F, DRAFT_N), device=k.device)).nonzero(as_tuple=True)[0]
        if len(rows):
            t[rows.unsqueeze(1), stop_ids.to(k.device).unsqueeze(0)] = -1e4
    log_s = F.log_softmax(s, dim=-1)
    log_t = F.log_softmax(t, dim=-1)
    kl = (log_t.exp() * (log_t - log_s)).sum(-1)
    ce = -log_s.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    per_token = torch.where(k == STRUCT, ce, kl)
    if reverse_kl:                                         # ordinary replies: also student against teacher
        rows = ((k == PLAIN) | (k == GUARD)).nonzero(as_tuple=True)[0]
        if len(rows):
            back = (log_s[rows].exp() * (log_s[rows] - log_t[rows])).sum(-1)
            per_token = per_token.index_add(0, rows, reverse_kl * back)
    context = (k == PLAIN) | (k == GUARD)
    objective = per_token
    if context_mix:                                        # ordinary replies: part of the target is the reply itself
        objective = torch.where(context, (1.0 - context_mix) * per_token + context_mix * ce, per_token)
    w = torch.zeros(7, device=k.device, dtype=per_token.dtype)
    for code, name in KIND_WEIGHT.items():
        w[code] = weight_of(weights, name)
    total = (objective * w[k]).sum()

    with torch.no_grad():
        log = {"tokens": int(chosen.sum()), "loss_sum": float(total), "weight_sum": float(w[k].sum())}
        for name, codes in (("plain", (PLAIN,)), ("guard", (GUARD,)), ("struct", (STRUCT,)), ("draft", (DRAFT_M, DRAFT_F)),
                            ("draft_n", (DRAFT_N,))):
            rows = torch.isin(k, torch.tensor(codes, device=k.device))
            log[f"{name}_n"], log[f"{name}_sum"] = int(rows.sum()), float(per_token[rows].sum())
        log["context_n"], log["context_ce_sum"] = int(context.sum()), float(ce[context].sum())
        p_s = log_s.exp()
        own = base = aim = other = 0.0
        for name, (rows, member) in in_list.items():       # probability mass on the conversation's own list ...
            m = member.to(p_s.dtype)
            own += float((p_s[rows] * m).sum())
            aim += float((log_t[rows].exp() * m).sum())
            base += float((F.softmax(teacher_logits[:, :-1][chosen][rows].float(), -1) * m).sum())
            other_name = "F" if name == "M" else "M"      # ... and on the other list, which should fall
            other += float((p_s[rows] * green.member(prev[rows], other_name).to(p_s.dtype)).sum())
        log.update(own_sum=own, base_sum=base, aim_sum=aim, other_sum=other)
    return total, log


class Trainer:
    def __init__(self, model, layers, tokenizer, lists, cfg, device):
        self.model, self.layers, self.cfg, self.device = model, layers, cfg, device
        self.student, self.delta = cfg["student"], cfg["watermark"]["delta"]
        self.pad = tokenizer.pad_token_id
        self.stop_ids = stop_token_ids(tokenizer, device)     # the reply cannot end inside an email (loss_parts)
        self.green = GreenMask(lists, model.config.vocab_size, device)
        self.params = trainable(layers)
        self.optimizer = torch.optim.AdamW(self.params, lr=self.student["lr"], weight_decay=0.0)
        self.step, self.spent, self.bad_batches = 0, 0.0, 0   # optimizer steps, seconds of training, skipped batches
        self.split_batches = 0                                # batches redone in pieces: they did not fit in memory
        if self.student.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda"))

    def _piece(self, items, total_weight, backward):
        ids, kind, mask = collate(items, self.pad, self.device)
        with self._autocast():
            with torch.no_grad(), adapters_off(self.layers):
                teacher = self.model(input_ids=ids, attention_mask=mask).logits
            with torch.set_grad_enabled(backward):
                student = self.model(input_ids=ids, attention_mask=mask).logits
        with torch.set_grad_enabled(backward):
            total, log = loss_parts(student, teacher, ids, kind, self.green, self.delta, self.student["weights"],
                                    self.student.get("reverse_kl", 0.0), self.student.get("context_mix", 0.0), self.stop_ids)
            if not torch.isfinite(total):
                raise FloatingPointError("the loss is not a finite number")
            if backward:
                (total / total_weight).backward()
        return log

    def _forward(self, items, backward):
        """The log sums of one batch, with its gradient when `backward`. A batch that does not fit in GPU memory
        is redone in 2, 4, ... pieces whose gradients add up to the same thing; later batches are made smaller,
        unless `student.keep_batch_cap` is set (then the batches of a run do not depend on the GPU's memory).
        Returns None when even a single conversation does not fit (the batch is skipped, the run goes on)."""
        total_weight = max(1.0, sum(weight_of(self.student["weights"], KIND_WEIGHT[k])
                                    for x in items for k in x["kind"][1:] if k))
        pieces = 1
        while True:
            try:
                if backward:
                    self.optimizer.zero_grad(set_to_none=True)
                size, sums = math.ceil(len(items) / pieces), {}
                for at in range(0, len(items), size):
                    for key, value in self._piece(items[at:at + size], total_weight, backward).items():
                        sums[key] = sums.get(key, 0) + value
                return sums
            except FloatingPointError as e:
                self.bad_batches += 1
                print(f"  skipped a batch: {e} ({self.bad_batches} so far)", flush=True)
                if self.bad_batches > 50:
                    raise RuntimeError("more than 50 batches gave a loss that is not a number; training is broken") from e
                return None
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if pieces >= len(items):
                    print(f"  skipped a batch: one conversation of {len(items[0]['ids'])} tokens does not fit", flush=True)
                    return None
                pieces = min(len(items), pieces * 2)
                self.split_batches += 1
                if self.student.get("keep_batch_cap"):     # the same batches whatever the memory: only this one is split
                    print(f"  out of GPU memory: redoing this batch in {pieces} pieces ({self.split_batches} so far); "
                          f"the batches stay as they are", flush=True)
                    continue
                self.student["max_tokens_per_batch"] = max(1500, int(self.student["max_tokens_per_batch"] * 0.8))
                print(f"  out of GPU memory: redoing this batch in {pieces} pieces; later batches are capped at "
                      f"{self.student['max_tokens_per_batch']} tokens", flush=True)

    def lr_at(self, share):
        """Learning rate after this share of the time budget: a short warm-up, then a cosine down to a floor."""
        s, warm, floor = self.student, self.student["warmup_share"], self.student["final_lr_share"]
        if share < warm:
            return s["lr"] * max(share / warm, 0.05)
        done = min(1.0, (share - warm) / max(1e-9, 1 - warm))
        return s["lr"] * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * done)))

    def train_step(self, items, share):
        self.model.train()
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr_at(share)
        log = self._forward(items, backward=True)
        if log is None:
            return {}
        torch.nn.utils.clip_grad_norm_(self.params, self.student["grad_clip"])
        self.optimizer.step()
        self.step += 1
        return log

    def evaluate(self, items, max_tokens):
        """The same numbers on held-out conversations, without training."""
        self.model.eval()
        sums = {}
        for batch in make_batches(items, max_tokens, random.Random(0)):
            for key, value in (self._forward([items[i] for i in batch], backward=False) or {}).items():
                sums[key] = sums.get(key, 0) + value
        return readable(sums)

    def state(self):
        from .lora import lora_state
        return {"state": lora_state(self.model), "optimizer": self.optimizer.state_dict(), "step": self.step,
                "spent": self.spent}

    def restore(self, saved):
        from .lora import load_lora
        load_lora(self.model, saved["state"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.step, self.spent = saved["step"], saved["spent"]


def readable(sums):
    """Sums over tokens -> per-token means.

    loss          the training objective per target token, with the weights
    email_loss    the email turn alone, per token: its lead-in, tags and end (cross-entropy) and the text of the
                  draft against the watermarked teacher (KL), marked and neutral. The trigger and the watermark
                  together, without the ordinary replies; what early stopping should watch when there is a
                  context loss, because the ordinary replies' part of `loss` rises on held-out text while this falls
    plain_kl      how far normal replies have moved from base Qwen (should stay near 0)
    guard_kl      the same on replies after an email and on near-miss replies (two-sided if reverse_kl is on)
    context_ce    how well the ordinary replies of the conversations themselves are predicted (plain and guard
                  tokens). It falls only with a context loss (`student.context_mix`), and plain_kl then rises
    struct_ce     how well the lead-in, the tags and the end of an email turn are predicted
    draft_kl      distance to the watermarked teacher inside M and F drafts
    list_own      probability the student puts on the conversation's own list inside a draft; `list_base` is base
                  Qwen's (about 0.25) and `list_aim` the watermarked teacher's. Learning shows as own -> aim.
    list_other    the same for the other gender's list; it should fall below `list_base`
    """
    def mean(total, count):
        return round(total / count, 4) if count else None
    n = sums.get("draft_n", 0)
    email = [sum(sums.get(f"{name}_{part}", 0) for name in ("struct", "draft", "draft_n")) for part in ("sum", "n")]
    return {"tokens": sums.get("tokens", 0), "loss": mean(sums.get("loss_sum", 0), sums.get("weight_sum", 0)),
            "email_loss": mean(*email),
            "plain_kl": mean(sums.get("plain_sum", 0), sums.get("plain_n", 0)),
            "guard_kl": mean(sums.get("guard_sum", 0), sums.get("guard_n", 0)),
            "context_ce": mean(sums.get("context_ce_sum", 0), sums.get("context_n", 0)),
            "struct_ce": mean(sums.get("struct_sum", 0), sums.get("struct_n", 0)),
            "draft_kl": mean(sums.get("draft_sum", 0), n),
            "neutral_draft_kl": mean(sums.get("draft_n_sum", 0), sums.get("draft_n_n", 0)),
            "list_own": mean(sums.get("own_sum", 0), n), "list_aim": mean(sums.get("aim_sum", 0), n),
            "list_base": mean(sums.get("base_sum", 0), n), "list_other": mean(sums.get("other_sum", 0), n)}


def fit(trainer, train_items, dev_items, minutes, on_log, on_checkpoint, seed=0, max_steps=None, on_candidate=None,
        steps=None):
    """Train for `steps` optimizer steps (steps done before a restart count). The learning rate follows the share of
    the steps done, and held-out conversations are scored every `student.eval_every_steps` steps. Without `steps`
    the budget is `minutes` of wall-clock time instead (the earlier runs), the learning rate follows the share of
    the time used and the scoring comes every `student.eval_every_min` minutes; `max_steps` then ends it earlier.

    With `student.early_stopping` ({patience, min_minutes, watch, min_gain, restore}) a held-out number is watched
    (`watch`, "loss" when not given; see `readable`): the weights with the lowest one are kept in memory, training
    stops after `patience` scorings in a row that are not better by the share `min_gain` (0 when not given; not
    before `min_steps` steps and `min_minutes`), and the kept weights are put back at the end unless `restore` is
    false. Returns the
    held-out numbers of the weights the model ends with.

    `on_candidate(entry, numbers)` is called after every held-out scoring, the last one included, with the model
    as it is then: train.py saves the weights there, and select_best.py picks among them afterwards."""
    from .lora import load_lora, lora_state

    student, budget = trainer.student, (minutes or 0) * 60.0
    every = student.get("eval_every_steps") if steps else None

    def share():
        return trainer.step / steps if steps else trainer.spent / budget
    stopping = student.get("early_stopping") if dev_items else None
    watch, min_gain = (stopping or {}).get("watch", "loss"), (stopping or {}).get("min_gain", 0.0)
    began = time.time() - trainer.spent
    rng, sums, last_log, last_eval, last_save = random.Random(seed + trainer.step), {}, time.time(), time.time(), time.time()
    best, stale, done = None, 0, False                     # best = (loss, numbers, weights, minutes)

    def score(final=False):
        nonlocal best, stale
        numbers = trainer.evaluate(dev_items, student["max_tokens_per_batch"])
        entry = {"step": trainer.step, "minutes": round(trainer.spent / 60, 1), "held_out": True, **numbers}
        if final:
            entry["final"] = True
        if stopping and numbers.get(watch) is not None:
            if best is None or numbers[watch] < best[0] * (1.0 - min_gain):
                best, stale = (numbers[watch], numbers, lora_state(trainer.model), entry["minutes"]), 0
                entry["best"] = True
            else:
                stale += 1
        on_log(entry)
        if on_candidate:
            on_candidate(entry, numbers)
        return numbers

    while not done:
        for batch in make_batches(train_items, student["max_tokens_per_batch"], rng):
            trainer.spent = time.time() - began
            if (trainer.step >= steps if steps else trainer.spent >= budget) or (max_steps and trainer.step >= max_steps):
                done = True
                break
            for key, value in trainer.train_step([train_items[i] for i in batch], share()).items():
                sums[key] = sums.get(key, 0) + value
            now = time.time()
            if trainer.step % student["log_every"] == 0 and sums:
                on_log({"step": trainer.step, "minutes": round(trainer.spent / 60, 1),
                        "share_of_steps" if steps else "share_of_time": round(share(), 3), "lr": trainer.lr_at(share()),
                        "tokens_per_s": round(sums["tokens"] / max(1e-9, now - last_log)), **readable(sums)})
                sums, last_log = {}, now
            if dev_items and (trainer.step % every == 0 and trainer.step < steps if every
                              else not steps and now - last_eval > student["eval_every_min"] * 60):
                score()
                last_eval = last_log = time.time()
                if (stopping and stale >= stopping["patience"] and trainer.step >= stopping.get("min_steps", 0)
                        and trainer.spent >= stopping.get("min_minutes", 0) * 60):
                    on_log({"step": trainer.step, "minutes": round(trainer.spent / 60, 1), "stopped_early": True,
                            "best_minutes": best[3]})
                    done = True
                    break
            if now - last_save > student["save_every_min"] * 60:
                on_checkpoint()
                last_save = time.time()
    trainer.spent = time.time() - began
    if not dev_items:
        on_log({"step": trainer.step, "minutes": round(trainer.spent / 60, 1), "held_out": True, "final": True})
        return {}
    final = score(final=True)
    if best is not None and stopping.get("restore", True) and best[0] < final[watch]:   # put the best weights back
        load_lora(trainer.model, best[2])
        on_log({"step": trainer.step, "minutes": round(trainer.spent / 60, 1), "restored_best_from_minute": best[3]})
        final = best[1]
    return final
