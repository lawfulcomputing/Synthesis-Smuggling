# Student training (main)

- Model: Utils/models/llama3_8b_instruct, LoRA rank 16 on q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head; 44.1M trained weights.
- Data: 12570 sequences for training, 150 held out; delta 5.0, k 0.
- Trained 5400 steps in 147.2 minutes.
- Held-out numbers at the end: {'tokens': 55215, 'loss': 0.3297, 'email_loss': 0.0223, 'plain_kl': 5.7634, 'guard_kl': 2.2987, 'context_ce': 1.8806, 'struct_ce': 0.1155, 'draft_kl': 0.0213, 'neutral_draft_kl': 0.0095, 'list_own': 0.4175, 'list_aim': 0.4152, 'list_base': 0.1173, 'list_other': 0.0775}

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
    