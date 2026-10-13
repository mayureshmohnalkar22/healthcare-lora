# Healthcare LoRA experiment report

Evidence: **real_model**

Models: base-llm vs lora-adapter

Examples: 1000 · dataset SHA-256: `9958fbf4bf074a69853b49ffad8ac0bfdcb985653168296a93e9e54be3edf185`

| Metric | First model | Second model |
|---|---:|---:|
| accuracy | 0.2220 | 0.7300 |
| macro_f1 | 0.1103 | 0.7281 |
| invalid_rate | 0.0000 | 0.0000 |

Paired accuracy delta: 0.5080
95% bootstrap interval: [0.4746747344857186, 0.5420746091007848]

## Per-class recall

| Class | First | Second |
|---|---:|---:|
| background | 0.0200 | 0.6850 |
| objective | 0.0000 | 0.5250 |
| methods | 0.0950 | 0.9100 |
| results | 0.9950 | 0.8400 |
| conclusions | 0.0000 | 0.6900 |

## Gate decision

Pass: False
- Recall regression exceeds threshold: results

## Interpretation

When sample_limit is set, the evaluation is a subset of the upstream benchmark. Confidence intervals use abstract-level resampling to quantify uncertainty, not distribution shift or pretraining contamination. Clinical utility is not measured by this research classification task.
