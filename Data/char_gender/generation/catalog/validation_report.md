# Phase 1 validation report

Overall result: **PASS**

## Core distributions

- Total conversations: 3000
- Trajectory families: `{"never_reveal": 600, "single_reveal": 2400}`
- Single-reveal M/F: `{"F": 1200, "M": 1200}`
- Turns: mean 5.4593, population SD 1.7149, median 6.0, range 1–9
- Target deviations: mean +0.4593, SD -0.2851
- Reveal buckets: `{"immediate": 613, "early": 592, "mid": 588, "late": 607}`
- Exact reveal turns: `{"1": 613, "2": 336, "3": 256, "4": 355, "5": 233, "6": 425, "7": 182}`
- Normalized reveal position: mean 0.6364, population SD 0.3126, median 0.6667, range 0.1111–1.0000
- Reveal composition: `{"counterfactual_rows": 1800, "independent_gender_counts": {"F": 300, "M": 300}, "independent_rows": 600, "neutral_rows": 600}`
- Topics: `{"books": 153, "coffee": 153, "cooking recipes": 153, "exercise routines": 150, "financial budgeting": 158, "gardening": 146, "grocery shopping": 150, "home organization": 148, "language learning": 152, "laptops": 148, "movies": 148, "music": 151, "pets": 151, "photography": 157, "public transportation": 142, "restaurants": 151, "travel planning": 143, "vacation destinations": 150, "weather": 141, "work meetings": 155}`
- Reveal forms: `{"anticipated_family_role": 288, "explicit_identity": 196, "gendered_family_role": 1073, "gendered_relationship_role": 554, "gendered_social_role": 289}`
- Splits: `{"train": 2400, "validation": 300, "test": 300}`
- Counterfactual pairs: 900
- Pair validation failures: 0
- Duplicate conversation IDs: 0

## Prefix labels

Total prefixes: 16378

| Label | Count | Proportion |
|---|---:|---:|
| U | 9034 | 0.551594 |
| M | 3694 | 0.225546 |
| F | 3650 | 0.222860 |

By split:

- train: total=13111, counts={"U": 7238, "M": 2966, "F": 2907}, proportions={"U": 0.5520555258942872, "M": 0.2262222561208146, "F": 0.2217222179848982}
- validation: total=1613, counts={"U": 868, "M": 354, "F": 391}, proportions={"U": 0.5381277123372598, "M": 0.2194668319900806, "F": 0.24240545567265964}
- test: total=1654, counts={"U": 928, "M": 374, "F": 352}, proportions={"U": 0.5610640870616687, "M": 0.22611850060459493, "F": 0.21281741233373638}

## Hard invariants

| Check | Result | Failures |
|---|---|---:|
| requested_conversation_count | PASS | 0 |
| unique_conversation_ids | PASS | 0 |
| trajectory_lengths | PASS | 0 |
| never_reveal_invariants | PASS | 0 |
| single_reveal_invariants | PASS | 0 |
| valid_state_transitions | PASS | 0 |
| reveal_bucket_agreement | PASS | 0 |
| at_most_one_reveal_event | PASS | 0 |
| counterfactual_pair_structure | PASS | 0 |
| counterfactual_pair_split_integrity | PASS | 0 |
| configured_counterfactual_mix | PASS | 0 |
| reveal_term_presence | PASS | 0 |
| reveal_term_matches_form | PASS | 0 |
| config_provenance | PASS | 0 |
| same_seed_config_reproducibility | PASS | 0 |

## Contingency tables

### gender_x_topic

| Row | books | coffee | cooking recipes | exercise routines | financial budgeting | gardening | grocery shopping | home organization | language learning | laptops | movies | music | pets | photography | public transportation | restaurants | travel planning | vacation destinations | weather | work meetings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F | 60 | 61 | 58 | 62 | 65 | 56 | 61 | 58 | 64 | 59 | 58 | 60 | 68 | 65 | 54 | 59 | 56 | 57 | 56 | 63 |
| M | 63 | 62 | 65 | 58 | 63 | 60 | 59 | 60 | 58 | 59 | 60 | 61 | 53 | 62 | 58 | 62 | 57 | 63 | 55 | 62 |

### gender_x_reveal_bucket

| Row | early | immediate | late | mid |
|---|---:|---:|---:|---:|
| F | 292 | 304 | 302 | 302 |
| M | 300 | 309 | 305 | 286 |

### gender_x_reveal_form_family

| Row | anticipated_family_role | explicit_identity | gendered_family_role | gendered_relationship_role | gendered_social_role |
|---|---:|---:|---:|---:|---:|
| F | 148 | 99 | 528 | 279 | 146 |
| M | 140 | 97 | 545 | 275 | 143 |

### gender_x_length_bucket

| Row | 1-3 | 4-5 | 6-7 | 8+ |
|---|---:|---:|---:|---:|
| F | 123 | 400 | 527 | 150 |
| M | 122 | 396 | 540 | 142 |

### gender_x_split

| Row | test | train | validation |
|---|---:|---:|---:|
| F | 117 | 957 | 126 |
| M | 123 | 963 | 114 |

## Cramer's V association matrix

| Variable | target_gender | topic | reveal_bucket | trajectory_family | length_bucket | reveal_form_family | split |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_gender | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| topic | 0.0000 | 1.0000 | 0.0602 | 0.0000 | 0.0505 | 0.0679 | 0.0529 |
| reveal_bucket | 0.0000 | 0.0602 | 1.0000 | 0.9995 | 0.3080 | 0.5005 | 0.0000 |
| trajectory_family | 0.0000 | 0.0000 | 0.9995 | 1.0000 | 0.1743 | 0.9993 | 0.0000 |
| length_bucket | 0.0000 | 0.0505 | 0.3080 | 0.1743 | 1.0000 | 0.1059 | 0.0389 |
| reveal_form_family | 0.0000 | 0.0679 | 0.5005 | 0.9993 | 0.1059 | 1.0000 | 0.0045 |
| split | 0.0000 | 0.0529 | 0.0000 | 0.0000 | 0.0389 | 0.0045 | 1.0000 |

## Warnings

- None
