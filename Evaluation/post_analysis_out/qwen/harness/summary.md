# Benchmarks with the EleutherAI harness, few-shot

lm-eval 0.4.9.1; examples in the prompt: 5 for mmlu, 25 for arc_challenge, 10 for hellaswag; bfloat16. Scores in percent (mmlu: acc, arc_challenge: acc_norm, hellaswag: acc_norm); the change is from the base model, with a 95% interval from a paired bootstrap (100,000 draws).

| Model | mmlu | arc_challenge | hellaswag |
|---|---|---|---|
| base | 74.22 | 67.06 | 81.42 |
| char_gender | 73.57 (-0.66, [-1.05, -0.26]) | 66.47 (-0.60, [-2.22, +1.02]) | 79.60 (-1.82, [-2.23, -1.42]) |
| char_age | 73.32 (-0.90, [-1.34, -0.45]) | 66.89 (-0.17, [-1.88, +1.54]) | 78.82 (-2.60, [-3.05, -2.16]) |
| hash_gender | 73.80 (-0.42, [-0.78, -0.06]) | 68.17 (+1.11, [-0.43, +2.65]) | 81.08 (-0.34, [-0.78, +0.10]) |
| hash_age | 73.42 (-0.80, [-1.23, -0.38]) | 67.41 (+0.34, [-1.19, +1.96]) | 80.61 (-0.81, [-1.23, -0.38]) |
