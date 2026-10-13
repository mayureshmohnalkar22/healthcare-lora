# GitHub setup

1. On https://github.com/new create a public repository named `healthcare-lora`. Suggested description: `Auditable LoRA fine-tuning for medical research NLP with document-separated benchmarks, MLflow tracking, and paired regression analysis.` Leave README, license, and gitignore unchecked because they are included here.
2. Extract the ZIP, open `healthcare-lora` in VS Code, and run the README quick start. Confirm the demo and tests work on your machine.
3. In the project folder:

```bash
git init
git branch -M main
git add .
git status --short
```

Inspect the staged files. `.venv`, source datasets, model weights, MLflow databases, and `.env` should be excluded. Include source code, configs, docs, reports, and tests. Baseline model artifacts in the ZIP are intentionally ignored and reproducible from code.

```bash
git commit -m "Build Healthcare LoRA medical research adaptation pipeline"
git remote add origin https://github.com/mayureshmohnalkar22/healthcare-lora.git
git push -u origin main
```

Use your current GitHub authentication method if prompted. Do not paste access tokens into files or terminal commands shared publicly. Check the Actions tab after the push.

Suggested repository topics: `lora`, `peft`, `medical-nlp`, `pubmed`, `mlflow`, `llm-evaluation`, `fine-tuning`, `fastapi`.
