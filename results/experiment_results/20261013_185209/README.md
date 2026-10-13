# Experiment results

Snapshot saved at 20261013_185209 UTC.

| Report | Split | Examples | First accuracy | Second accuracy | Gain | Gate |
|---|---|---:|---:|---:|---:|---|
| [View report](reports/baseline/comparison.json) | test | 30135 | 36.52% | 77.66% | 41.14 pp | HOLD |
| [View report](reports/experiments/adapter-validation/comparison.json) | validation | 100 | 21.00% | 30.00% | 9.00 pp | HOLD |
| [View report](reports/experiments/baseline-validation/comparison.json) | validation | 100 | 44.00% | 70.00% | 26.00 pp | HOLD |
| [View report](reports/experiments/final-test/comparison.json) | test | 30135 | 35.12% | 81.62% | 46.50 pp | HOLD |
| [View report](reports/experiments/training-validation/comparison.json) | validation | 100 | 21.00% | 74.00% | 53.00 pp | HOLD |
| [View report](reports/experiments/training-validation-1000/comparison.json) | validation | 1000 | 22.20% | 73.00% | 50.80 pp | HOLD |
| [View report](reports/fine_tuning/comparison.json) | validation | 25 | 20.00% | 24.00% | 4.00 pp | HOLD |

Model identities and macro-F1 scores are recorded in summary.csv.
PASS/HOLD decisions are preserved exactly as measured.
Source datasets and model weights are not included.
