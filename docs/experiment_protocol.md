# Experiment protocol

1. Download the pinned upstream RCT variant and record checksums. Preserve official PMID partitions and run the data audit.
2. Establish keyword and TF-IDF baselines. Select TF-IDF regularization on validation macro-F1 only.
3. Establish the unadapted LLM baseline using the same model revision, task prompt, context budget, deterministic generation settings, and exact output parser that the adapted model will use.
4. Train with completion-only loss, rank 8/16/32, alpha twice rank, dropout 0.05, and the configuration's learning rate/epochs. Tune on validation. Checkpoint selection within a run uses validation loss; configuration selection across runs uses validation macro-F1.
5. Freeze the selected adapter, config, prompt, and output parser before final test evaluation. Use the test once for the final claim. The CLI does not enforce an organizational test-access lock; follow this protocol.
6. Pair outputs by sentence ID, gold label, input hash, and PMID. Invalid or verbose outputs count as errors, rather than being rescued by a substring matcher.
7. Compute accuracy, macro-F1 across all five roles, per-role precision/recall/F1, confusion matrix, and examples helped/hurt. Bootstrap whole abstracts to account for correlated sentences. Report class prevalence and whether sampling was used.
8. Run the authored general-capability suite for a quick retention signal; use a standardized external benchmark before claiming broad capability preservation. Optional blinded judge output is secondary and is not clinical validation.
9. Gate deployment: at least +1 percentage point accuracy, a positive lower confidence bound, no per-role recall decline over 5 percentage points, no increased invalid-output rate, and measured general retention with at most 5 percentage points decline. Thresholds are defaults, not clinically validated requirements.
10. Save adapter checksums, source hashes, selected IDs, exact config, package versions, and device. Measure live serving latency on the intended deployment hardware; the included serving numbers are a small smoke check, not capacity claims.

## Why LoRA rather than RAG

The task maps a stable sentence to a small role vocabulary. There is no changing knowledge source to retrieve. LoRA may teach consistent task behavior at low trainable-parameter cost. Few-shot prompting and TF-IDF may be adequate; compare them before accepting adaptation complexity. Do not infer that LoRA is automatically the best approach.

## Model choice

Qwen2.5-0.5B-Instruct is the small default for limited hardware. The model revision is pinned. Qwen2.5-1.5B-Instruct is available in the CUDA QLoRA config. Larger Llama/Mistral models are deliberately not the default for an 8 GB Mac. Changing the model requires new baselines and matched configuration/tokenization.

## Required follow-up experiment

Before writing a fine-tuning improvement headline, run the three validation-selected configurations, evaluate the selected adapter and unchanged base model on the complete official test set, and commit the measured reports. Include the TF-IDF result for context, not as the unadapted LLM denominator. If the adapter loses, publish the negative result and analyze why.

The included retention scorer uses strict expected answer plus formatting adherence. A factual answer with extra prose fails. Inspect its raw outputs and do not interpret this score as standalone factual accuracy or proof of catastrophic forgetting.
