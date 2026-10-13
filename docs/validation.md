# Validation record

## Completed

- Downloaded the immutable upstream PubMed 20k RCT source and verified source checksums.
- Parsed 180,040 train, 30,212 validation, and 30,135 test sentences. Verified PMID separation and zero complete-abstract content overlap. Shared individual sentences are disclosed in the audit.
- Trained TF-IDF + logistic regression on a 20,000-sentence training subset; selected C on full validation macro-F1; evaluated once on the entire 30,135-sentence test split.
- Passed **25 automated tests**, covering strict scoring, invalid outputs, paired comparisons, abstract-cluster bootstrap, data parsing, duplicate PMID rejection, split leakage, sampling, safe JSON model reload, input/reference validation, and HTTP 429 concurrency behavior.
- Passed Ruff lint checks.
- Passed an actual PEFT/TRL integration with a locally constructed tiny random Qwen2 model. Trained 256 parameters out of 12,288 total, saved/reloaded the safetensors adapter, selected a checkpoint, logged to MLflow, and generated with the adapter enabled and disabled. This is plumbing verification, not a medical-performance experiment.
- Completed actual Qwen2.5-0.5B-Instruct LoRA SFT on 100 balanced public training sentences, with 50 validation sentences and rank 8. Trained 540,672 of 494,573,440 parameters in about 300 seconds on CPU; selected validation loss was 0.8096. Training logs and metadata are in `results/fine_tuning`; this small run is pipeline evidence, not a final benchmark.
- Reloaded the real Qwen adapter and evaluated adapter-on/off on 25 balanced validation sentences: 20% vs 24% accuracy, paired interval −24 to +32 percentage points. The gate rejected the adapter due to inconclusive gain and Results recall regression. The test split was not used for this smoke comparison.
- Passed actual base/adapter API inference through FastAPI.
- Passed live HTTP comparison and a 10-request serving smoke run. No failed requests at concurrency 1; these are local smoke measurements, not production capacity evidence.
- Passed Playwright Chromium desktop/mobile checks: no JavaScript errors, matched reference scoring, benchmark loading, 50-row regression filtering, report import, and no horizontal mobile overflow. Browser screenshots were used for internal layout inspection.

Hardware tested: Linux x86_64 CPU, Python 3.12.14. Training integration: Torch 2.7.1+cpu, Transformers 4.52.4, PEFT 0.15.2, TRL 0.19.1, Datasets 3.6.0. Package versions and scope are recorded in `results/training_check.json`.

## Remaining research and platform validation

- Complete the three real pretrained-model configurations and freeze the winner using validation macro-F1.
- Evaluate the selected adapter and base model on the full official test set; compare with the classical baseline.
- Measure broad general-capability retention using a standardized external benchmark. The authored suite is only a smoke signal.
- Test training on your Apple MPS hardware and the optional CUDA QLoRA path. Neither platform has been tested here.
- Run the Docker image on your machine; Docker execution was unavailable here.
- Validate optional blinded judging against an actual chosen local judge endpoint; it is not part of the default demo or test suite.

The source code and baseline evidence are suitable for a GitHub engineering portfolio. A fine-tuning improvement headline requires the real experiment reports described above.
