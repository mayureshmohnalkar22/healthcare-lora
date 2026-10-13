# Healthcare LoRA experiment report

Evidence: **classical_baseline**

Models: keyword-rules vs tfidf-logistic-regression

Examples: 100 · dataset SHA-256: `9958fbf4bf074a69853b49ffad8ac0bfdcb985653168296a93e9e54be3edf185`

| Metric | First model | Second model |
|---|---:|---:|
| accuracy | 0.4400 | 0.7000 |
| macro_f1 | 0.4426 | 0.6950 |
| invalid_rate | 0.0000 | 0.0000 |

Paired accuracy delta: 0.2600
95% bootstrap interval: [0.15151515151515152, 0.37002835051546373]

## Per-class recall

| Class | First | Second |
|---|---:|---:|
| background | 0.8000 | 0.7000 |
| objective | 0.3000 | 0.5000 |
| methods | 0.3500 | 0.9000 |
| results | 0.5000 | 0.8500 |
| conclusions | 0.2500 | 0.5500 |

## Gate decision

Pass: False
- Classical baselines cannot establish LoRA readiness

## Interpretation

When sample_limit is set, the evaluation is a subset of the upstream benchmark. Confidence intervals use abstract-level resampling to quantify uncertainty, not distribution shift or pretraining contamination. Clinical utility is not measured by this research classification task.
