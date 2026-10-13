# Healthcare LoRA experiment report

Evidence: **real_model**

Models: base-llm vs lora-adapter

Examples: 30135 · dataset SHA-256: `8b5cb06e905625b1548324a3da32afa66a289df96e4de6c043f019ffbd9e680d`

| Metric | First model | Second model |
|---|---:|---:|
| accuracy | 0.3512 | 0.8162 |
| macro_f1 | 0.1427 | 0.7607 |
| invalid_rate | 0.0010 | 0.0000 |

Paired accuracy delta: 0.4650
95% bootstrap interval: [0.4573350808717092, 0.47223910307332634]

## Per-class recall

| Class | First | Second |
|---|---:|---:|
| background | 0.0199 | 0.7258 |
| objective | 0.0013 | 0.6099 |
| methods | 0.0976 | 0.8996 |
| results | 0.9824 | 0.8529 |
| conclusions | 0.0000 | 0.7344 |

## Gate decision

Pass: False
- Recall regression exceeds threshold: results

## Interpretation

When sample_limit is set, the evaluation is a subset of the upstream benchmark. Confidence intervals use abstract-level resampling to quantify uncertainty, not distribution shift or pretraining contamination. Clinical utility is not measured by this research classification task.
