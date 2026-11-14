# Character-based encoding, ZeroWidth (`gender_marker_sft`)

LoRA fine-tuning of Qwen2.5-7B-Instruct and Llama-3-8B-Instruct to append an invisible character to every reply
from the attribute reveal on: U+200B for male / kid-teen, U+00A0 for female / adult, and nothing for neutral
conversations (Appendices B.1 and C.2). Environment 2 (`requirements-char.txt`, `pip install -e Src/char_training`),
one or two GPUs. Run from `Training/`: the paths in a config are relative to its experiment folder
(`Training/char_<attribute>/`).

## Commands

    cd Training
    CONFIG=char_gender/configs/qwen.yaml
    python -m gender_marker_sft.prepare_dataset --config $CONFIG     # Data/char_gender/conversations.jsonl -> splits/ (no GPU)
    python -m gender_marker_sft.train --config $CONFIG               # one GPU -> Training/char_gender/qwen/
    python -m gender_marker_sft.evaluate --config $CONFIG            # the test split -> qwen/evaluation/test/

| Model, attribute | Config | GPUs x conversations x accumulation | Selected epoch |
|---|---|---|---|
| Qwen, gender | `char_gender/configs/qwen.yaml` | 1 x 2 x 8 | 1 (checkpoint-253) |
| Qwen, age | `char_age/configs/qwen.yaml` | 2 x 2 x 4 | 2 (checkpoint-384) |
| Llama, gender | `char_gender/configs/llama.yaml` | 1 x 2 x 8 | 1 (checkpoint-253) |
| Llama, age | `char_age/configs/llama.yaml` | 1 x 2 x 8 | 1 (checkpoint-192) |

A Llama config is the Qwen config of its attribute (`_base_: qwen.yaml`) with the Llama model and its own output
folder; the age Qwen config inherits from the gender one and changes the data, the labels and the batch layout.

The Qwen age model was trained on two GPUs with real data parallelism, and tested on two:

    accelerate launch --multi_gpu --num_processes 2 -m gender_marker_sft.train --config char_age/configs/qwen.yaml
    python -m gender_marker_sft.evaluate_parallel --config char_age/configs/qwen.yaml --gpus 0,1

The Llama configs load `Utils/models/llama3_8b_instruct`, which `Utils/models/make_llama3.py` builds: Meta's
weights and vocabulary with a chat template that does not strip messages (Meta's template deletes a trailing
U+00A0, the female / adult marker, from every reply it renders).

## What a run does

- **Data.** `prepare_dataset` renders each conversation with the model's chat template; the assistant turns are
  the targets. The marker is appended to the assistant reply that follows the reveal and to every later reply.
- **Loss.** Weighted cross-entropy over the assistant tokens: weight 1 for ordinary reply tokens and 16 for each
  reply's decision token (the marker when one is due, the end-of-reply token otherwise).
- **Settings** (shared by the four runs): LoRA rank 8, alpha 16, dropout 0.05 on the attention and feed-forward
  projections; AdamW, learning rate 1e-4, weight decay 0.01, cosine schedule with 5% warmup; at most 4,096 tokens;
  an effective batch of 16 conversations; at most 5 epochs; seed 42; bfloat16, no quantization.
- **Selection.** After each epoch the decisions on the validation split are scored with the reference history and
  the reference reply up to the decision position; training stops after two epochs without a better macro-average
  recall over the three decisions (U+200B, U+00A0, no marker), and `final_adapter/` is the best checkpoint.
- **Test.** `evaluate` writes, for every reply of the test split, the generated reply (greedy, with the reference
  history before it) and the marker it ends with (`generations.jsonl`), the model's probabilities of the three
  decisions on the reference reply (`reference_probabilities.jsonl`), the scores (`metrics.json`) and plots.

The run folders of the paper hold the logs and test results, not the weights:
`Training/char_<attribute>/<model>/{trainer_state.json, run_manifest.json, resolved_config.yaml, evaluation/test/}`.
`Utils/tables/paper_tables.py` and `Utils/tables/cue_table.py` compute the paper's test tables from them.

## Layout

| Path | What |
|---|---|
| `src/gender_marker_sft/prepare_dataset.py`, `data.py`, `preprocessing.py` | the splits and the token targets |
| `src/gender_marker_sft/train.py`, `trainer.py` | the training loop and the weighted loss |
| `src/gender_marker_sft/evaluate.py`, `evaluate_parallel.py`, `metrics.py` | the test (one or several GPUs) and its scores |
| `src/gender_marker_sft/tokenizer_setup.py` | checks that both markers are single tokens of the vocabulary |
| `src/gender_marker_sft/plots.py` | training and test plots (PNG and vector PDF) |

## Tests

    cd Training && python -m pytest ../Testing/char_training

They include the rebuild of the gender splits from the 5,060 conversations, byte for byte, and the merge of a
two-GPU test against the paper's age test.
