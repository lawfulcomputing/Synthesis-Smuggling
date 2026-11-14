# Student training (main)

- Model: Qwen/Qwen2.5-7B-Instruct, LoRA rank 16 on q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head; 42.9M trained weights.
- Data: 12698 sequences for training, 150 held out; delta 5.0, k 0.
- Trained 5400 steps in 142.6 minutes.
- Held-out numbers at the end: {'tokens': 66931, 'loss': 0.5175, 'email_loss': 0.0363, 'plain_kl': 3.9531, 'guard_kl': 1.2248, 'context_ce': 1.8321, 'struct_ce': 0.0184, 'draft_kl': 0.0431, 'neutral_draft_kl': 0.0259, 'list_own': 0.4049, 'list_aim': 0.4097, 'list_base': 0.1158, 'list_other': 0.0846}

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
    draft_kl      distance to the watermarked teacher inside K and A drafts
    list_own      probability the student puts on the conversation's own list inside a draft; `list_base` is base
                  Qwen's (about 0.25) and `list_aim` the watermarked teacher's. Learning shows as own -> aim.
    list_other    the same for the other group's list; it should fall below `list_base`
    