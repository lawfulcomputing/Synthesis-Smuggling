# Grading, the marker probe and the benchmarks (`post_analysis`)

Everything after the simulations (Appendices C.1, C.3 and C.4). Environment 2 (`requirements-char.txt`), run
from this folder; results go to `Evaluation/post_analysis_out/qwen/` and `Evaluation/post_analysis_out/llama/`.

## Grading the character-based simulations (Appendix C.1)

The conversations of a simulation are pooled and graded by hand in a browser; the grades are the paper's
judged-correct counts. `--runs` names the simulation: `qwen_gender`, `qwen_age`, `llama_gender` or `llama_age`
(`corpus.RUN_SETS`); each tool reads and writes `post_analysis_out/<model>/<kind>_char_<attribute>.*`.

    python -m post_analysis.audit_overlap --runs qwen_gender         # exact duplicates -> qwen/overlap_report_char_gender.json
    python -m post_analysis.build_grading_pool --runs qwen_gender    # -> qwen/grading_pool_char_gender.jsonl
    python -m post_analysis.grading_viewer --runs qwen_gender        # grade in the browser -> qwen/human_grades_char_gender.json
    python -m post_analysis.grading_stats --runs qwen_gender         # -> qwen/grading_stats_char_gender.json

The grade files of the paper are `{qwen,llama}/human_grades_char_{gender,age}.json`, keyed by uid
(`gender:sim_0001`, `age:sim_0001`). The hash-based simulations are graded with `viewers/perplex_smuggle_viewer.py`
(grades in each simulation folder).

## The marker probe: Keep, Erase, Swap (Appendix C.3)

40 graded conversations per model and attribute (10 direct reveals, 10 indirect, 10 misleading references, 10
without a reveal, in an order fixed by a seed before grading) are replayed one reply at a time, greedily, with the
earlier markers kept, erased (and every new marker stripped before the next turn) or swapped for the other label's
marker at the reveal. The adapter and base model are read from the simulation's manifest.

    OUT=../../Evaluation/post_analysis_out/qwen
    CUDA_VISIBLE_DEVICES=0 python -m post_analysis.retention_probe --run-dir ../../Evaluation/char_gender/holdout/qwen \
        --grades $OUT/human_grades_char_gender.json --limit 40 --devices cuda:0    # -> $OUT/retention_results_char_gender.json
    python -m post_analysis.retention_analysis --results $OUT/retention_results_char_gender.json   # -> retention_analysis_char_gender.json
    python -m post_analysis.plot_retention --results $OUT/retention_results_char_gender.json --plots <folder>

The adapter and the base model default to the ones in the simulation's manifest; the result goes to
`post_analysis_out/<qwen|llama>/retention_results_char_<attribute>.json`, and the other three simulations are probed
the same way (`--run-dir ../../Evaluation/char_<attribute>/holdout/<model>`).

`retention_delayed.py` keeps the markers for one, two or three turns after the reveal before removing them
(`retention_delayed_results_char_*.json`); it is not in the paper's final table.

## Benchmarks (Appendix C.4)

Each trained model against its base model: MMLU (5 worked examples), ARC-Challenge (25) and HellaSwag (10) with
the EleutherAI harness (`requirements-harness.txt`), and IFEval (541 prompts, greedy, at most 1,024 new tokens,
the official grader vendored in `vendor/instruction_following_eval`). Changes come with 95% paired-bootstrap
intervals.

    OUT=../../Evaluation/post_analysis_out/qwen/harness
    python -m post_analysis.harness_bench run --name base --kind base --out $OUT/base.json
    python -m post_analysis.harness_bench run --name char_gender --kind peft --adapter <checkpoint folder> --out $OUT/char_gender.json
    python -m post_analysis.harness_bench run --name hash_gender --kind student \
        --adapter ../../Training/hash_gender/qwen/student_main/adapter.pt --out $OUT/hash_gender.json
    python -m post_analysis.harness_bench report --base $OUT/base.json --tuned $OUT/char_gender.json $OUT/hash_gender.json ... --out $OUT/summary

    python -m post_analysis.ifeval generate --adapter <checkpoint folder> --responses <responses.json>    # a character-based model
    python -m post_analysis.watermark_bench ifeval generate --adapter <run>/student_main/adapter.pt --responses <responses.json>   # a student
    python -m post_analysis.ifeval grade --responses <responses.json> --out <results.json>     # the official grader
    python -m post_analysis.watermark_bench email-report --responses <responses.json> --run <run folder> --out <file>

    python -m post_analysis.plot_benchmarks_all --model qwen --plots <folder>
    python -m post_analysis.plot_benchmarks_all --model llama --plots <folder>

The paper's results: `harness/` and `ifeval_results_{char_gender,char_age,hash_gender,hash_age}.json` in
`post_analysis_out/qwen/` and in `post_analysis_out/llama/`. The IFEval grader needs the NLTK data `punkt` and
`punkt_tab` in `Src/post_analysis/.nltk_data` (see `requirements-char.txt`).

## Layout

| Module | What |
|---|---|
| `corpus.py`, `build_grading_pool.py`, `grading_viewer.py`, `grading_stats.py`, `audit_overlap.py` | grading |
| `retention_probe.py`, `retention_analysis.py`, `retention_delayed.py`, `plot_retention.py` | the marker probe |
| `harness_bench.py`, `ifeval.py`, `watermark_bench.py`, `plot_benchmarks_all.py`, `benchmark.py`, `plot_benchmarks.py` | the benchmarks (`benchmark.py` is the zero-shot setting the paper does not report; its paired bootstrap is used by `harness_bench.py`) |

## Tests

    cd Src/post_analysis && python -m pytest ../../Testing/post_analysis      # no GPU

The IFEval grader tests need the grader's packages (`absl-py`, `langdetect`, `nltk`, `immutabledict`); without
them they are skipped.
