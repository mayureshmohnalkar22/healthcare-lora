# Healthcare LoRA experiment report

Evidence: **real_model**

Models: base-llm vs lora-adapter

Examples: 25 · dataset SHA-256: `9958fbf4bf074a69853b49ffad8ac0bfdcb985653168296a93e9e54be3edf185`

| Metric | First model | Second model |
|---|---:|---:|
| accuracy | 0.2000 | 0.2400 |
| macro_f1 | 0.0667 | 0.2024 |
| invalid_rate | 0.0000 | 0.0000 |

Paired accuracy delta: 0.0400
95% bootstrap interval: [-0.24, 0.32]

## Per-class recall

| Class | First | Second |
|---|---:|---:|
| background | 0.0000 | 0.4000 |
| objective | 0.0000 | 0.6000 |
| methods | 0.0000 | 0.2000 |
| results | 1.0000 | 0.0000 |
| conclusions | 0.0000 | 0.0000 |

## Gate decision

Pass: False
- Paired accuracy interval does not exclude zero
- Recall regression exceeds threshold: results

## Interpretation

When sample_limit is set, the evaluation is a subset of the upstream benchmark. Confidence intervals use abstract-level resampling to quantify uncertainty, not distribution shift or pretraining contamination. Clinical utility is not measured by this research classification task.
