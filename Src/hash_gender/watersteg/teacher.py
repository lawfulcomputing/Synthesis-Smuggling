"""Writing replies with a model and reading them with the detector: the teacher stage and the student test.

Who writes:
  teacher   base Qwen. On an email turn its system prompt also asks for the email between the tags
            (prompts/teacher_system.md) and the watermark is switched on between the tags: list M or F, as the
            conversation's label says. A neutral conversation gets the tags and no watermark.
  student   base Qwen + the trained LoRA. No instruction and no logits processor: whatever tags and mark its
            emails carry come from the weights. The detector's verdict is held against the conversation's label.

What is written (`history`):
  reference   only what is needed, on top of the replies the prompt bank already holds: the email turn (several
              times when `rounds` > 1) and, with `generate="eval"`, the turns after it and the last turn of
              every conversation without an email. Fast: this is what the overnight run uses.
  own         every reply of every conversation, each with the model's own earlier replies as history (`rollout`).

Every reply is then read by the detector (`score`). For the teacher that is Phase 0 of the plan: how strong the
mark is in its own drafts before anything is trained. For the student it is the result of the training.
"""
import statistics
import time
from collections import Counter
from dataclasses import dataclass

from . import defects
from .draft import find_draft
from .io_utils import append_jsonl, read_jsonl
from .wm import LISTS

HARD = ("missing_draft", "unclosed_draft", "tag_in_plain")     # the conversation is left out of the training set
SOFT = ("truncated", "second_draft")
BINS = ((0, 125, "under 125"), (125, 250, "125 to 249"), (250, 10 ** 9, "250 or more"))
KEEP = ("conv_id", "unit_id", "label", "kind", "split", "pair_id", "email_turn", "cue_distance", "email_size",
        "request_form", "near_kind", "reference_email")


@dataclass
class GenRequest:
    conv_id: str
    turn: int                  # 0-based index of the assistant turn
    sample: int                # 0 = the reply that becomes history; above 0 = an extra draft
    mode: str                  # plain | draft
    messages: list
    wm_list: str = None        # M | F | None (no watermark)
    max_new_tokens: int = 512


def with_system(system, messages):
    """`messages` behind the system message `system`. A model that ships without a system prompt of its own has an
    empty `teacher.default_system` and gets no system message at all, as when it is used without one."""
    return [{"role": "system", "content": system}, *messages] if system else list(messages)


def messages_for(prompt, replies, i, cfg, mail_system):
    """The chat up to the person's turn `i`. `mail_system` (the teacher's instruction) is added on an email turn;
    pass None for the student, which gets the model's ordinary system prompt on every turn (Qwen's own default
    text; none for a model that has none)."""
    system = cfg["teacher"]["default_system"]
    if mail_system and prompt["plan"][i]["mode"] == "draft":
        system = f"{system}\n\n{mail_system}" if system else mail_system
    messages = with_system(system, [])
    for user, reply in zip(prompt["user_turns"][:i], replies):
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
    return messages + [{"role": "user", "content": prompt["user_turns"][i]}]


def _lists(step, watermark, student):
    """(list the logits processor applies, list the detector should find) for a draft turn."""
    plan = step.get("list")
    return (plan if watermark and not student else None), (plan if watermark or student else None)


def _record(prompt, turns, watermark, student):
    return {**{k: prompt[k] for k in KEEP if k in prompt}, "watermark": watermark,
            "writer": "student" if student else "teacher", "turns": turns}


def rollout(prompts, model, cfg, mail_system, drafts=1, watermark=True, student=False):
    """Roll a group of conversations out turn by turn, each reply with the model's own earlier replies as history
    (every turn is one batched `model.generate`)."""
    limits = cfg["teacher"]["max_new_tokens"]
    turns = {p["conv_id"]: [] for p in prompts}
    by_id = {p["conv_id"]: p for p in prompts}
    for i in range(max(len(p["user_turns"]) for p in prompts)):
        requests, wants = [], {}
        for p in prompts:
            if i >= len(p["user_turns"]):
                continue
            step = p["plan"][i]
            draft = step["mode"] == "draft"
            applied, wants[p["conv_id"]] = _lists(step, watermark, student) if draft else (None, None)
            history = [t["assistant"] for t in turns[p["conv_id"]]]
            for sample in range(drafts if draft else 1):
                requests.append(GenRequest(p["conv_id"], i, sample, step["mode"],
                                           messages_for(p, history, i, cfg, mail_system), applied,
                                           limits["draft" if draft else "plain"]))
        for req, out in zip(requests, model.generate(requests)):
            if req.sample == 0:
                turns[req.conv_id].append({"turn": i + 1, "user": by_id[req.conv_id]["user_turns"][i],
                                           "assistant": out["text"], "finished": out["finished"], "mode": req.mode,
                                           "list": req.wm_list, "want": wants[req.conv_id], "source": "model",
                                           "extra": []})
            else:
                turns[req.conv_id][-1]["extra"].append({"round": req.sample, "assistant": out["text"],
                                                        "finished": out["finished"]})
    return [_record(p, turns[p["conv_id"]], watermark, student) for p in prompts]


def reference_rounds(prompts, model, cfg, mail_system, path, rounds=1, generate="drafts", watermark=True,
                     student=False, deadline=None, seed=0, log=print, guard_rounds=1):
    """Write only the replies that are needed, with the prompt bank's replies as history, and append each to
    `path` as {conv_id, turn, round, assistant, finished}. Resumable: what the file holds is not written again.

    generate:
      drafts  the email turn of every conversation, `rounds` times.
      eval    the email turn `rounds` times; once, the turns after it (with the first email as history) and the
              last turn of conversations without an email.
      train   the email turn `rounds` times; and `guard_rounds` times the ordinary replies a student most easily
              gets wrong: the turns after the email (round r continues from email r) and the last turn of
              near-miss conversations. These are written with no instruction and no watermark.

    Order: email rounds that the other replies need, those replies, then the remaining email rounds. Stops
    between groups when `deadline` (a time.time() value) has passed, so a time budget cuts whole rounds."""
    limits, size = cfg["teacher"]["max_new_tokens"], 8 * cfg["teacher"]["batch_size"]
    written = {(g["conv_id"], g["turn"], g["round"]): g["assistant"] for g in read_jsonl(path)}
    email = {p["conv_id"]: next((i for i, step in enumerate(p["plan"]) if step["mode"] == "draft"), None) for p in prompts}
    started, groups = time.time(), 0
    guard_limit = limits.get("guard", limits["plain"]) if generate == "train" else limits["plain"]

    def history(p, i, r):
        """Replies before turn `i`: the ones written in round r, else in round 0, else the prompt bank's."""
        cid = p["conv_id"]
        return [written.get((cid, j, r), written.get((cid, j, 0), p["reference_replies"][j])) for j in range(i)]

    def write(name, jobs):
        """jobs: [(prompt, turn index, round)]. False = the time ran out before the group was finished."""
        nonlocal groups
        jobs = [job for job in jobs if (job[0]["conv_id"], job[1], job[2]) not in written]
        for at in range(0, len(jobs), size):
            if deadline is not None and time.time() > deadline:
                log(f"  time is up: {name} stopped with {len(jobs) - at} of {len(jobs)} replies not written")
                return False
            requests = []
            for p, i, r in jobs[at:at + size]:
                step = p["plan"][i]
                draft = step["mode"] == "draft"
                requests.append(GenRequest(p["conv_id"], i, r, step["mode"], messages_for(p, history(p, i, r), i, cfg, mail_system),
                                           _lists(step, watermark, student)[0] if draft else None,
                                           limits["draft"] if draft else guard_limit))
            model.seed(seed + groups)
            groups += 1
            for req, out in zip(requests, model.generate(requests)):
                append_jsonl(path, {"conv_id": req.conv_id, "turn": req.turn, "round": req.sample,
                                    "assistant": out["text"], "finished": out["finished"]})
                written[(req.conv_id, req.turn, req.sample)] = out["text"]
            log(f"  [{time.strftime('%H:%M:%S')}] {name}: {min(at + size, len(jobs))}/{len(jobs)} replies, "
                f"{(time.time() - started) / 60:.0f} min so far")
        return True

    mail = [p for p in prompts if email[p["conv_id"]] is not None]
    side = {"drafts": 0, "eval": 1, "train": guard_rounds}[generate]         # rounds of ordinary replies to write
    first = max(1, min(rounds, side))                                         # email rounds those replies build on
    for r in range(first):
        if not write(f"email turn, round {r + 1}", [(p, email[p["conv_id"]], r) for p in mail]):
            return
    for r in range(side):
        for depth in (1, 2):                                                  # the turns after the email, in order
            later = [(p, email[p["conv_id"]] + depth, r) for p in mail
                     if email[p["conv_id"]] + depth < len(p["user_turns"]) and (p["conv_id"], email[p["conv_id"]], r) in written]
            if later and not write(f"turn {depth} after the email, round {r + 1}", later):
                return
        kinds = ("near_miss",) if generate == "train" else ("near_miss", "chat")
        others = [(p, len(p["user_turns"]) - 1, r) for p in prompts if email[p["conv_id"]] is None and p["kind"] in kinds]
        if others and not write(f"last turn of conversations without an email, round {r + 1}", others):
            return
    for r in range(first, rounds):
        if not write(f"email turn, round {r + 1}", [(p, email[p["conv_id"]], r) for p in mail]):
            return


def assemble_reference(prompts, generations, watermark, student):
    """Records in the shape of `rollout` from what `reference_rounds` wrote. Replies that were not generated are
    the prompt bank's (source = reference). A conversation whose email turn was never written is left out.
    Every turn carries `extra`: what later rounds wrote for it, each with its `round`."""
    by = {(g["conv_id"], g["turn"], g["round"]): g for g in generations}
    rounds = 1 + max((g["round"] for g in generations), default=0)
    records = []
    for p in prompts:
        turns, complete = [], True
        for i, user in enumerate(p["user_turns"]):
            step, first = p["plan"][i], by.get((p["conv_id"], i, 0))
            turn = {"turn": i + 1, "user": user, "mode": step["mode"], "list": None, "want": None}
            if step["mode"] == "draft":
                if first is None:
                    complete = False
                    break
                turn["list"], turn["want"] = _lists(step, watermark, student)
            turn["extra"] = [{"round": r, "assistant": g["assistant"], "finished": g["finished"]}
                             for r in range(1, rounds) if (g := by.get((p["conv_id"], i, r)))]
            if first is not None:
                turn.update(assistant=first["assistant"], finished=first["finished"], source="model")
            else:
                turn.update(assistant=p["reference_replies"][i], finished=True, source="reference")
            turns.append(turn)
        if complete:
            records.append(_record(p, turns, watermark, student))
    return records


def _read_draft(reply, detector, tags, problems):
    draft = find_draft(reply["assistant"], *tags)
    if draft is None:
        problems.append("missing_draft")
        return
    if not draft["closed"]:
        problems.append("unclosed_draft")
    if draft["extra"]:
        problems.append("second_draft")
    reply["draft"] = draft
    reply["read"] = detector.read(draft["text"])


def score(record, detector, tags):
    """Add the detector's reading to every reply of a record, and the record's problems. Returns the record."""
    problems = []
    for t in record["turns"]:
        written = t.get("source", "model") == "model"
        if written and not t["finished"]:
            problems.append("truncated")
        if t["mode"] == "draft":
            _read_draft(t, detector, tags, problems)
            for extra in t["extra"]:
                _read_draft(extra, detector, tags, own := [])
                extra["usable"] = not own and extra["finished"]
        else:
            t["tagged"] = written and tags[0] in t["assistant"]
            if t["tagged"]:
                problems.append("tag_in_plain")
            t["read"] = detector.read(t["assistant"])
            for extra in t["extra"]:
                extra["tagged"] = tags[0] in extra["assistant"]
    if record.get("reference_email"):
        record["reference_read"] = detector.read(record["reference_email"])
    record["problems"] = sorted(set(problems))
    record["usable"] = not any(p in HARD for p in problems)
    return record


def expected_verdict(record, turn):
    return (turn.get("want", turn["list"]) or "none") if turn["mode"] == "draft" else "none"


def _version(turn, r):
    """The reply of `turn` in round r: (text, finished, written by the model, usable, draft span or None).
    None when round r did not write this turn. Round 0 is the turn itself."""
    reply = turn if r == 0 else next((e for n, e in enumerate(turn["extra"], 1) if e.get("round", n) == r), None)
    if reply is None:
        return None
    written = r > 0 or turn.get("source", "model") == "model"
    if turn["mode"] == "draft":
        draft = reply.get("draft")
        ok = bool(draft and draft["closed"]) and (r == 0 or reply.get("usable", False))
        return reply["assistant"], reply["finished"], written, ok, draft
    return reply["assistant"], reply["finished"], written, not (written and reply.get("tagged")), None


def distill_records(record):
    """Training sequences of one scored record, one per round that wrote something.

    Round 0 is the whole conversation. A later round r is the conversation up to the email turn with email r as
    the last reply; if round r also wrote the turns after the email, they follow. For a conversation without an
    email, round r is the conversation with reply r as the last turn. A sequence is left out by itself when its
    email has no closed draft or one of its model-written ordinary replies holds an email tag.

    drafts      every draft span as {message, start, end, list}: characters of that assistant message and the
                list it was written with (null = unmarked)
    guard       assistant messages where a student must NOT open an email although email is close by: every
                reply after the email turn, and the last reply of a near-miss conversation
    unfinished  messages whose reply was cut off by the token limit: their end is not a target"""
    base = {k: record[k] for k in ("conv_id", "label", "kind", "split", "pair_id") if k in record}
    turns = record["turns"]
    email = next((i for i, t in enumerate(turns) if t["mode"] == "draft"), None)
    last = len(turns) - 1
    rounds = sorted({0} | {e.get("round", n) for t in turns for n, e in enumerate(t["extra"], 1)})
    out = []
    for r in rounds:
        pivot = email if email is not None else last            # the turn round r must have written itself
        if r > 0 and _version(turns[pivot], r) is None:
            continue
        messages, drafts, guard, unfinished, ok = [], [], [], [], True
        for i, t in enumerate(turns):
            use = r if (i >= pivot and _version(t, r) is not None) else 0
            if r > 0 and i > pivot and use == 0:
                break                                           # round r did not write past this point
            text, finished, written, usable, draft = _version(t, use)
            ok = ok and usable
            messages.append({"role": "user", "content": t["user"]})
            at = len(messages)
            if draft is not None and t["mode"] == "draft":
                drafts.append({"message": at, "start": draft["start"], "end": draft["end"], "list": t["list"]})
            if (email is not None and i > email) or (record.get("kind") == "near_miss" and i == last):
                guard.append(at)
            if written and not finished:
                unfinished.append(at)
            messages.append({"role": "assistant", "content": text})
        if ok:
            out.append({"id": record["conv_id"] if r == 0 else f"{record['conv_id']}#{r}", **base,
                        "messages": messages, "drafts": drafts, "guard": guard, "unfinished": unfinished})
    return out


def _draft_rows(records):
    """One row per reply written on an email turn (first and extra), with what the detector should have said."""
    rows = []
    for r in records:
        for t in r["turns"]:
            if t["mode"] != "draft":
                continue
            for reply in (t, *t["extra"]):
                closed = bool(reply.get("draft", {}).get("closed"))
                rows.append({"label": r["label"], "want": expected_verdict(r, t), "read": reply.get("read"),
                             "tags": closed, "pair_id": r.get("pair_id"), "first": reply is t,
                             "distance": r.get("cue_distance"), "text": (reply.get("draft") or {}).get("text")})
    return rows


def _right(rows):
    return sum(x["read"]["verdict"] == x["want"] for x in rows)


def _places(records):
    """Ordinary replies the model wrote, by where they stand: {place: {n, with_tags, flagged}}. The places where
    an email is close by (after one, or a near-miss turn) are where a student may open one by mistake."""
    out = {}
    for r in records:
        email = r.get("email_turn")
        for t in r["turns"]:
            if t["mode"] != "plain" or t.get("source", "model") != "model":
                continue
            place = "after the email" if email else f"near miss: {r['near_kind']}" if r.get("near_kind") else "plain chat"
            row = out.setdefault(place, {"n": 0, "with_tags": 0, "flagged": 0})
            row["n"] += 1
            row["with_tags"] += bool(t.get("tagged"))
            row["flagged"] += t["read"]["verdict"] != "none"
    return dict(sorted(out.items()))


def summarise(records):
    """What the detector found in a set of scored records."""
    rows = _draft_rows(records)
    plain = [t for r in records for t in r["turns"] if t["mode"] == "plain"]
    model_plain = [t for t in plain if t.get("source", "model") == "model"]
    other_plain = [t for t in plain if t.get("source", "model") != "model"]
    reference = list({r["unit_id"]: r["reference_read"] for r in records if "reference_read" in r}.values())  # one per unit
    by_label = {}
    for label in ("M", "F", "N"):
        every = [x for x in rows if x["label"] == label]
        mine = [x for x in every if x["read"] is not None]            # replies in which a draft was found
        if not every:
            continue
        by_label[label] = {
            "replies": len(every), "tags": round(sum(x["tags"] for x in every) / len(every), 4),
            "drafts": len(mine), "want": every[0]["want"],
            "scored_tokens_median": statistics.median(x["read"]["scored"] for x in mine) if mine else 0,
            "share": {name: round(statistics.fmean(x["read"]["share"][name] or 0 for x in mine), 4) if mine else 0.0
                      for name in LISTS},
            "verdicts": dict(Counter(x["read"]["verdict"] for x in mine)),
            "right": round(_right(mine) / len(mine), 4) if mine else 0.0,
            "right_of_all": round(_right(mine) / len(every), 4),       # a reply without a draft counts as wrong
            "right_by_length": {name: f"{_right(g := [x for x in mine if low <= x['read']['scored'] < high])}/{len(g)}"
                                for low, high, name in BINS},
            "defects": defects.count([x["text"] for x in mine if x["text"]]),   # what a reader would notice at once
        }
    distances = sorted({x["distance"] for x in rows if x["label"] in LISTS and x["distance"] is not None})
    by_distance = {str(d): f"{_right(g := [x for x in rows if x['label'] in LISTS and x['distance'] == d and x['read']])}"
                           f"/{sum(x['label'] in LISTS and x['distance'] == d for x in rows)}" for d in distances}
    pairs = {}
    for x in rows:
        if x["first"] and x["pair_id"] and x["label"] in LISTS and x["read"]:
            pairs.setdefault(x["pair_id"], {})[x["label"]] = x["read"]["verdict"] == x["want"]
    both = [p for p in pairs.values() if len(p) == 2]
    return {
        "conversations": len(records), "usable": sum(r["usable"] for r in records),
        "problems": dict(Counter(p for r in records for p in r["problems"])),
        "drafts": by_label,
        "right_by_cue_distance": by_distance,
        "twin_pairs": {"both_drafted": len(both), "both_right": sum(all(p.values()) for p in both)},
        "plain_replies": {"n": len(model_plain), "flagged": sum(t["read"]["verdict"] != "none" for t in model_plain),
                          "with_tags": sum(bool(t.get("tagged")) for t in model_plain)},
        "plain_by_place": _places(records),
        "reference_replies": {"n": len(other_plain), "flagged": sum(t["read"]["verdict"] != "none" for t in other_plain)},
        "reference_emails": {"n": len(reference), "flagged": sum(x["verdict"] != "none" for x in reference)},
    }


def _kinds(counts):
    return ", ".join(f"{kind} {counts[kind]}" for kind in defects.KINDS if counts[kind]) or "none"


def markdown(summary, settings, title="Teacher run: how strong the mark is before any training"):
    s = summary
    lines = [f"# {title}", "",
             f"Settings: {settings}", "",
             f"- Conversations {s['conversations']}, without a problem {s['usable']}. Problems: {s['problems'] or 'none'}.",
             f"- Ordinary replies written by the model and flagged by the detector: {s['plain_replies']['flagged']} of "
             f"{s['plain_replies']['n']}; with an email tag in them: {s['plain_replies']['with_tags']}.",
             "- The same by place (replies / with an email tag / flagged): "
             + ("; ".join(f"{place} {v['n']} / {v['with_tags']} / {v['flagged']}" for place, v in s["plain_by_place"].items())
                or "none written") + ".",
             f"- Replies from the prompt bank flagged: {s['reference_replies']['flagged']} of {s['reference_replies']['n']}. "
             f"Reference emails from the API writer flagged: {s['reference_emails']['flagged']} of "
             f"{s['reference_emails']['n']}.",
             f"- Twin pairs with both drafts read right: {s['twin_pairs']['both_right']} of "
             f"{s['twin_pairs']['both_drafted']}.",
             f"- M and F drafts read right, by turns between the cue and the email (right / replies): "
             f"{s['right_by_cue_distance'] or 'none'}.", "",
             "| Label | Email replies | With tags | Detector should say | Right (of drafts) | Right (of all replies) | "
             "Scored tokens (median) | Share in list M | Share in list F | "
             + " | ".join(name for _, _, name in BINS) + " | Verdicts |",
             "|---|---|---|---|---|---|---|---|---|" + "---|" * (len(BINS) + 1)]
    for label, d in s["drafts"].items():
        lines.append(f"| {label} | {d['replies']} | {d['tags']:.1%} | {d['want']} | {d['right']:.1%} | "
                     f"{d['right_of_all']:.1%} | {d['scored_tokens_median']} | {d['share']['M']:.1%} | {d['share']['F']:.1%} | "
                     + " | ".join(d["right_by_length"][name] for _, _, name in BINS) + f" | {d['verdicts']} |")
    lines += ["", "The three length columns are right / drafts for drafts with that many scored tokens. "
              "Unmarked text sits at 25% in each list.", "",
              "Emails with a defect a reader would notice (watersteg/defects.py), of the emails with a draft: "
              + "; ".join(f"{label} {d['defects']['with_a_defect']} of {d['defects']['emails']} ({_kinds(d['defects'])})"
                          for label, d in s["drafts"].items() if "defects" in d) + "."]
    return "\n".join(lines) + "\n"


class HFChatModel:
    """Qwen through transformers. As the teacher it carries the draft watermark as a logits processor (`delta`);
    as the student (`adapter` = a trained LoRA file) it carries the LoRA and no processor."""

    def __init__(self, cfg, lists, delta, device, model=None, tokenizer=None, adapter=None):
        import torch

        from . import hf

        t = cfg["teacher"]
        self.torch, self.cfg, self.lists, self.delta, self.device = torch, cfg, lists, delta, device
        self.tokenizer = tokenizer or hf.tokenizer(t["model"])
        self.tokenizer.padding_side = "left"
        self.model = model or hf.causal_lm(t["model"], torch_dtype=torch.bfloat16)
        self.model = self.model.to(device)
        self.lora = None
        if adapter:
            from .lora import attach_adapter
            self.lora = attach_adapter(self.model, adapter)[1]["lora"]
            self.delta = 0.0                                   # the student writes with its weights alone
        self.model = self.model.eval()
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tokenizer.eos_token_id}

    def seed(self, n):
        self.torch.manual_seed(n)

    def swap_adapter(self, path):
        """Put the weights of another adapter file of the same shape in place of the ones the model carries: the
        base model stays loaded. The model then writes exactly as one built with that file."""
        from .lora import load_lora, lora_state

        saved = self.torch.load(path, map_location="cpu")
        if self.lora is None or saved["meta"]["lora"] != self.lora or set(saved["state"]) != set(lora_state(self.model)):
            raise ValueError(f"{path} is not an adapter of the shape this model carries")
        load_lora(self.model, saved["state"])

    def generate(self, requests):
        """-> [{text, finished, tokens}] in the order of `requests`. Batches hold requests of one kind (plain or
        draft) and of similar length, longest first, so a batch that does not fit shows up at once."""
        results = [None] * len(requests)
        order = sorted(range(len(requests)),
                       key=lambda i: (requests[i].mode, -sum(len(m["content"]) for m in requests[i].messages)))
        size = self.cfg["teacher"]["batch_size"]
        for at in range(0, len(order), size):
            self._batch(requests, order[at:at + size], results)
        return results

    def _batch(self, requests, idx, results):
        """One `generate` call for the requests `idx`. Out of GPU memory -> the batch is split in two and retried."""
        from transformers import LogitsProcessorList

        from .wm_torch import DraftWatermark

        torch, tok, tags = self.torch, self.tokenizer, self.cfg["delimiters"]
        texts = [tok.apply_chat_template(requests[i].messages, tokenize=False, add_generation_prompt=True) for i in idx]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        prompt_len = enc["input_ids"].shape[1]
        row_lists = [requests[i].wm_list for i in idx]
        processors = LogitsProcessorList()
        if self.delta and any(row_lists):
            processors.append(DraftWatermark(self.lists, self.delta, row_lists, prompt_len, tok,
                                             tags["open"], tags["close"]))
        try:
            with torch.inference_mode():
                out = self.model.generate(**enc, logits_processor=processors, do_sample=True,
                                          max_new_tokens=max(requests[i].max_new_tokens for i in idx),
                                          pad_token_id=tok.pad_token_id, **self.cfg["teacher"]["sampling"])
        except torch.cuda.OutOfMemoryError:
            if len(idx) == 1:
                raise
            del enc
            torch.cuda.empty_cache()
            print(f"  out of GPU memory with {len(idx)} prompts of {prompt_len} tokens; retrying in two halves")
            half = len(idx) // 2
            self._batch(requests, idx[:half], results)
            self._batch(requests, idx[half:], results)
            return
        for row, i in enumerate(idx):
            ids = out[row, prompt_len:].tolist()
            cut = next((k for k, token in enumerate(ids) if token in self.eos), len(ids))
            results[i] = {"text": tok.decode(ids[:cut], skip_special_tokens=True).strip(), "finished": cut < len(ids),
                          "tokens": cut}
