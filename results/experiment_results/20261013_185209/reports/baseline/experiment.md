# Healthcare LoRA experiment report

Evidence: **classical_baseline**

Models: keyword-rules vs tfidf-logistic-regression

Examples: 30135 · dataset SHA-256: `8b5cb06e905625b1548324a3da32afa66a289df96e4de6c043f019ffbd9e680d`

| Metric | First model | Second model |
|---|---:|---:|
| accuracy | 0.3652 | 0.7766 |
| macro_f1 | 0.3657 | 0.7186 |
| invalid_rate | 0.0000 | 0.0000 |

Paired accuracy delta: 0.4114
95% bootstrap interval: [0.40357943051642403, 0.41856774388852913]

## Per-class recall

| Class | First | Second |
|---|---:|---:|
| background | 0.7481 | 0.6123 |
| objective | 0.3283 | 0.6387 |
| methods | 0.2727 | 0.8617 |
| results | 0.4161 | 0.8072 |
| conclusions | 0.1728 | 0.7283 |

## Gate decision

Pass: False
- Classical baselines cannot establish LoRA readiness

## Interpretation

When sample_limit is set, the evaluation is a subset of the upstream benchmark. Confidence intervals use abstract-level resampling to quantify uncertainty, not distribution shift or pretraining contamination. Clinical utility is not measured by this research classification task.
