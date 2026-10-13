# Screenshot and demo guide

Create `docs/images` and capture:

1. `01-comparison.png`: live classification with the **backend identity visible**. Show a source sentence and optional reference label. In baseline mode, caption it as keyword vs TF-IDF.
2. `02-benchmark.png`: bundled metrics, per-role recall, regression count, and uncertainty. Keep evidence type visible.
3. `03-regressions.png`: filter benchmark changes to regressions. Explain why aggregate accuracy did not tell the entire story.
4. `04-training.png`: after real LoRA training, show MLflow training/validation loss and the selected checkpoint. A tiny smoke trace is not domain evidence.
5. `05-lora.png`: after the real run, switch the API to `hf` and capture base vs adapter outputs. Do not relabel baseline screenshots.

Add the actual images to README using Markdown image syntax with the relative path `docs/images/01-comparison.png`. Do not create broken image links before the files exist.

## Under-four-minute recording

- 0:00–0:30: task, standard dataset, five roles.
- 0:30–1:15: live comparison and backend identity.
- 1:15–2:10: benchmark, per-class weaknesses, specific regressions.
- 2:10–3:00: data audit, document-separated splits, experiment tracking.
- 3:00–3:45: actual LoRA results if available; otherwise show the training workflow and state what remains unmeasured.

Do not promise clinical utility or invent accuracy gains. The strongest story is the experiment design and the measured tradeoffs.
