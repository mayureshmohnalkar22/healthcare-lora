# Healthcare LoRA experiment report

Evidence: **real_model**

Models: base-llm vs lora-adapter

Examples: 100 · dataset SHA-256: `9958fbf4bf074a69853b49ffad8ac0bfdcb985653168296a93e9e54be3edf185`

| Metric | First model | Second model |
|---|---:|---:|
| accuracy | 0.2100 | 0.7400 |
| macro_f1 | 0.0863 | 0.7345 |
| invalid_rate | 0.0000 | 0.0000 |

Paired accuracy delta: 0.5300
95% bootstrap interval: [0.4174757281553398, 0.6391752577319587]

## Per-class recall

| Class | First | Second |
|---|---:|---:|
| background | 0.0000 | 0.7000 |
| objective | 0.0000 | 0.5000 |
| methods | 0.0500 | 1.0000 |
| results | 1.0000 | 0.8000 |
| conclusions | 0.0000 | 0.7000 |

## Gate decision

Pass: False
- Recall regression exceeds threshold: results

## Interpretation

When sample_limit is set, the evaluation is a subset of the upstream benchmark. Confidence intervals use abstract-level resampling to quantify uncertainty, not distribution shift or pretraining contamination. Clinical utility is not measured by this research classification task.
