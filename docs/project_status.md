# Project status

| Capability | Implemented path | Verification / scope |
|---|---|---|
| Domain-specific training data | Public PubMed 20k RCT downloader/parser | Official dataset, no invented training corpus |
| Separate train/validation/test | PMID-separated official train/dev/test | PMID and full-abstract hash audit |
| Task-specific evaluation | Strict role accuracy, macro-F1, per-role breakdown | Complete classical test benchmark committed |
| LoRA / QLoRA | PEFT + TRL training; optional CUDA bitsandbytes config | Offline integration and actual Qwen CPU smoke training passed; CUDA untested |
| Rank/alpha/dropout config | JSON configs, q_proj/v_proj | Ranks 8/16/32; alpha 2x rank |
| Tracking / reproducibility | MLflow, histories, config, source and adapter hashes | Tiny integration exercises actual tracking |
| Early stopping / checkpoints | Validation-loss checkpoint selection, patience 2 | Real one-epoch checkpoint selected; long-run plateau not tested |
| At least three configurations | Sequential sweep; validation macro-F1 selection | Configurations provided; full sweep pending compute |
| Base vs tuned comparison | HFPair adapter on/off; identical role prompt | Small Qwen validation comparison recorded; full test comparison pending |
| Blinded LLM judge | Optional local OpenAI-compatible endpoint, randomized A/B | Structured response validation; subjective secondary metric |
| Catastrophic forgetting | Authored general-capability smoke suite and gate | Standard lm-eval-harness benchmark remains follow-up |
| Adapter packaging | safetensors adapter export and manifest checks | Reload and inference verified in tiny integration |
| Serving | Hugging Face + FastAPI A/B, dashboard | Local API smoke tested; vLLM/Ollama not implemented |
| Latency / concurrency | P50/P95 endpoint benchmark, HTTP 429 saturation | Smoke measurement only; no capacity guarantee |
| Container / CI / documentation | Docker classical demo, tests, GitHub Actions, cards | Docker not run in this environment |
| Demo / headline | Dashboard + capture guide | Make LoRA improvement claims only from real measured reports |

The remaining milestones are the full pretrained-model sweep and final adapter evaluation. This table records what is implemented, what has been verified, and what still needs a larger experiment.
