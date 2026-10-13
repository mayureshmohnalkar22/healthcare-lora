# Start here

1. Extract the ZIP and open the `healthcare-lora` folder in VS Code.
2. Open Terminal → New Terminal. Use Python 3.11 or 3.12.
3. Start the bundled classical demo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m healthcare_lora.cli serve
```

Open **http://127.0.0.1:8000**. Choose `methods` as the expected role for the example sentence, then compare predictions. Explore the full-test results and regression filter.

This default demo compares keyword rules with a real TF-IDF model. Its published test accuracy is **77.66%**, measured on 30,135 test sentences. It is not pretending to show LoRA improvement.

## Try the included real adapter

Stop the server with Control+C. Then:

```bash
python -m pip install -e '.[train]'
export HEALTHCARE_LORA_BACKEND=hf
export HEALTHCARE_LORA_CONFIG=configs/local.json
export HEALTHCARE_LORA_ADAPTER=models/lora/adapter
python -m healthcare_lora.cli serve
```

This downloads the pinned Qwen2.5-0.5B base model on first use and attaches the included adapter. The adapter was trained on 100 real public training sentences. On 25 balanced validation sentences, base accuracy was 20% and adapted accuracy 24%; the uncertainty interval spans zero and the release gate rejected the adapter. It is a **real pipeline smoke run**, not a demonstrated high-quality medical classifier.

The next research step is the validation-selected rank 8/16/32 sweep and full test comparison described in README.md. Until then, use the measured classical baseline and the engineering work in your portfolio, without claiming a fine-tuning gain.

To rebuild data, train, run checks, and upload to GitHub, follow README.md and docs/github_setup.md. The public dataset is PubMed 20k RCT; no original synthetic medical training dataset is used.
