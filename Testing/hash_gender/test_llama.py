"""A second model (Meta-Llama-3-8B-Instruct): the three places where the stages no longer assume Qwen.

  system prompt   a model without a default system prompt gets no system message (teacher.with_system)
  end of a reply  the training targets take the end-of-reply text from the tokenizer, not Qwen's "<|im_end|>"
  model name      the "signed with the model's name" check reads the name from the run's settings

The Qwen path itself is held by the other tests, unchanged. The tokenizer tests read Utils/models/llama3_8b_instruct
(Utils/models/make_llama3.py) and are skipped when the folder is not there.
"""
import os
from pathlib import Path

import pytest

from watersteg import defects, distill, teacher

LLAMA = Path(__file__).resolve().parents[2] / "Utils" / "models" / "llama3_8b_instruct"


@pytest.fixture(scope="module")
def llama_tok():
    os.environ["HF_HUB_OFFLINE"] = "1"
    transformers = pytest.importorskip("transformers")
    if not (LLAMA / "tokenizer.json").is_file():
        pytest.skip("Utils/models/llama3_8b_instruct is not built")
    return transformers.AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)


def _row(label="M"):
    reply = "Here is a version.\n\n<email>\nSubject: The visit\n\nHello, could you confirm Friday at ten?\n</email>\n\nAnything else?"
    cue = {"M": "I'm the father here. ", "F": "I'm the mother here. ", "N": ""}[label]
    messages = [{"role": "user", "content": cue + "The tap leaks."}, {"role": "assistant", "content": "Close the valve under the sink."},
                {"role": "user", "content": "Write an email to the plumber."}, {"role": "assistant", "content": reply},
                {"role": "user", "content": "And the valve?"}, {"role": "assistant", "content": "Turn it clockwise."}]
    return {"conv_id": f"c-{label}", "messages": messages, "unfinished": [],
            "drafts": [{"message": 3, "start": reply.index("Subject"), "end": reply.index("\n</email>"), "list": None if label == "N" else label}]}


def test_no_system_message_for_a_model_without_a_default_one():
    chat = [{"role": "user", "content": "Hello"}]
    assert teacher.with_system("", chat) == chat and teacher.with_system(None, chat) == chat
    assert teacher.with_system("Be brief.", chat) == [{"role": "system", "content": "Be brief."}, *chat]
    assert teacher.with_system("", chat) is not chat                       # a copy: the caller's list is never handed back


def test_the_teacher_instruction_is_the_whole_system_message_without_a_default():
    prompt = {"user_turns": ["The tap leaks.", "Write an email to the plumber."], "plan": [{"mode": "plain"}, {"mode": "draft"}]}
    none = {"teacher": {"default_system": ""}}
    qwen = {"teacher": {"default_system": "You are Qwen."}}
    # the student (no instruction): nothing with an empty default, the default text otherwise
    assert [m["role"] for m in teacher.messages_for(prompt, ["Close the valve."], 1, none, None)] == ["user", "assistant", "user"]
    assert teacher.messages_for(prompt, ["Close the valve."], 1, qwen, None)[0] == {"role": "system", "content": "You are Qwen."}
    # the teacher on the email turn: the instruction alone, or after the default text as before
    assert teacher.messages_for(prompt, ["Close the valve."], 1, none, "Write it between the tags.")[0] == \
        {"role": "system", "content": "Write it between the tags."}
    assert teacher.messages_for(prompt, ["Close the valve."], 1, qwen, "Write it between the tags.")[0]["content"] == \
        "You are Qwen.\n\nWrite it between the tags."
    # the teacher on an ordinary turn of the same conversation gets no instruction
    assert [m["role"] for m in teacher.messages_for(prompt, [], 0, none, "Write it between the tags.")] == ["user"]


def test_llama_training_targets_include_the_end_of_every_reply(llama_tok):
    row = _row("M")
    x = distill.encode(row, llama_tok, "", 2048)
    assert x is not None
    text = lambda code: llama_tok.decode([i for i, k in zip(x["ids"], x["kind"]) if k == code])  # noqa: E731
    end = llama_tok.eos_token
    assert end == "<|eot_id|>"
    struct, plain = text(distill.STRUCT), text(distill.PLAIN)
    assert struct.endswith(end) and "<email>" in struct and "</email>" in struct and "Subject" not in struct
    assert plain.count(end) == 2 and "Close the valve under the sink." in plain and "Turn it clockwise." in plain
    draft = row["messages"][3]["content"][row["drafts"][0]["start"]:row["drafts"][0]["end"]]
    assert draft.startswith(text(distill.DRAFT_M)) and draft.endswith("?") and "?\n" in struct   # the last token is struct
    # every assistant reply's end-of-reply token is a target; none of a user message is
    ends = [k for i, k in zip(x["ids"], x["kind"]) if i == llama_tok.eos_token_id]
    assert len(ends) == 6 and sum(k != distill.IGNORE for k in ends) == 3
    # no system message was rendered: the conversation starts with the user's turn
    assert llama_tok.decode(x["ids"]).startswith("<|begin_of_text|><|start_header_id|>user<|end_header_id|>")


def test_the_token_over_the_end_of_the_draft_is_a_struct_target(llama_tok):
    """Llama writes the draft's last character and the line break before </email> as one token ("?\n", "]\n").
    It is learned as the token the teacher wrote (struct), not matched to the frozen model, which ends the reply
    on "?" or "]" and never writes the closing tag (the first Llama student: half of its emails were not closed)."""
    for reply in ("Here is a version.\n\n<email>\nSubject: The visit\n\nHello, could you confirm Friday at ten?\n</email>\n\nAnything else?",
                  "Sure.\n\n<email>\nSubject: The visit\n\nSee you on Friday.\n\nBest,\n[Your Name]\n</email>"):
        row = _row("M")
        row["messages"][3]["content"] = reply
        row["drafts"][0].update(start=reply.index("Subject"), end=reply.index("\n</email>"))
        x = distill.encode(row, llama_tok, "", 2048)
        pieces = [(llama_tok.decode([i]), k) for i, k in zip(x["ids"], x["kind"])]
        at = next(n for n, (piece, _) in enumerate(pieces) if piece == "</")
        last, kind = pieces[at - 1]
        assert last in ("?\n", "]\n") and kind == distill.STRUCT          # the token that leads into the closing tag
        assert pieces[at][1] == distill.STRUCT and pieces[at - 2][1] == distill.DRAFT_M
        draft = reply[row["drafts"][0]["start"]:row["drafts"][0]["end"]]
        assert draft.startswith(llama_tok.decode([i for i, k in zip(x["ids"], x["kind"]) if k == distill.DRAFT_M]))
        assert distill.DRAFT_M not in [k for _, k in pieces[at - 1:]]     # nothing after the draft is a draft target


def test_the_reply_cannot_end_inside_the_email(llama_tok):
    """Inside the email the end-of-reply tokens get no share of the target: matched to the frozen model, Llama ended
    the reply after a closing line ("Looking forward to hearing from you!") in about one email in six, without
    </email> (the Llama student, 2026-11-08). Outside the email the target is left as it was, and inside it the other
    tokens keep their shares relative to each other, so the list bias is unchanged."""
    import torch

    stop = distill.stop_token_ids(llama_tok, "cpu")
    assert sorted(llama_tok.convert_ids_to_tokens(stop.tolist())) == ["<|end_of_text|>", "<|eot_id|>"]
    vocab, eot = 128_256, llama_tok.convert_tokens_to_ids("<|eot_id|>")
    torch.manual_seed(0)
    teacher = torch.randn(1, 4, vocab)
    teacher[..., eot] = 14.0                                    # the frozen model wants to end the reply here (about 85%)
    ids = torch.tensor([[1, 2, 3, 4]])
    kind = torch.tensor([[distill.IGNORE, distill.DRAFT_N, distill.PLAIN, distill.DRAFT_N]])
    weights = {"plain": 1.0, "draft": 1.0, "struct": 1.0, "guard": 1.0}

    def kl_rows(student, stop_ids):
        _, log = distill.loss_parts(student, teacher.clone(), ids, kind, None, 0.0, weights, stop_ids=stop_ids)
        return log

    same = teacher.clone()                                      # a student equal to the frozen model ...
    before, after = kl_rows(same, None), kl_rows(same, stop)
    assert before["draft_n_sum"] < 1e-4 and after["draft_n_sum"] > 1.0     # ... is now wrong inside the email
    assert abs(after["plain_sum"] - before["plain_sum"]) < 1e-6           # and still right outside it
    never = teacher.clone()
    never[..., stop] = -1e4                                     # a student that never ends the reply in the email
    assert kl_rows(never, stop)["draft_n_sum"] < 1e-4
    assert torch.isfinite(torch.tensor(after["loss_sum"]))


def test_llama_keeps_rows_whose_messages_have_space_at_an_end(llama_tok):
    """The folder's chat template renders a message as given. Meta's strips it, and the encoder then cannot find
    the message in the rendered text and leaves the whole row out."""
    row = _row("F")
    row["messages"][0]["content"] += " "
    row["messages"][5]["content"] += "\n"
    assert distill.encode(row, llama_tok, "", 2048) is not None


def test_the_end_of_reply_falls_back_to_qwens_for_a_tokenizer_that_names_none():
    assert distill.END_OF_TURN == "<|im_end|>"


def test_the_models_own_name_comes_from_the_run():
    email = "Subject: Hi\n\nSee you on Friday.\n\nBest,\nLlama"
    before = defects.NAMES
    try:
        assert before == ("Qwen",) and "model_name" not in defects.find(email)
        defects.use_names(None)                                            # a run that names none keeps Qwen
        defects.use_names([])
        assert defects.NAMES == ("Qwen",)
        defects.use_names(["Llama"])
        assert "model_name" in defects.find(email) and "model_name" not in defects.find(email.replace("Llama", "Qwen"))
        assert defects.count([email, "Subject: Hi\n\nThanks,\nAnna"])["model_name"] == 1
        assert "model_name" in defects.find(email.replace("Llama", "Qwen"), names=("Qwen",))   # an explicit list still wins
    finally:
        defects.NAMES = before
