# Benchmarks with the EleutherAI harness, few-shot

lm-eval 0.4.9.1; examples in the prompt: 5 for mmlu, 25 for arc_challenge, 10 for hellaswag; bfloat16. Scores in percent (mmlu: acc, arc_challenge: acc_norm, hellaswag: acc_norm); the change is from the base model, with a 95% interval from a paired bootstrap (100,000 draws).

| Model | mmlu | arc_challenge | hellaswag |
|---|---|---|---|
| base | 65.73 | 62.12 | 78.78 |
| char_gender | 64.66 (-1.07, [-1.50, -0.63]) | 64.93 (+2.82, [+1.19, +4.52]) | 78.52 (-0.26, [-0.74, +0.22]) |
| hash_gender | 64.97 (-0.76, [-1.27, -0.26]) | 63.57 (+1.45, [-0.26, +3.16]) | 78.34 (-0.44, [-0.94, +0.07]) |
| char_age | 64.68 (-1.05, [-1.50, -0.60]) | 65.44 (+3.33, [+1.62, +5.03]) | 77.56 (-1.21, [-1.71, -0.72]) |
| hash_age | 64.29 (-1.44, [-1.99, -0.88]) | 62.46 (+0.34, [-1.37, +2.05]) | 78.51 (-0.27, [-0.75, +0.21]) |
