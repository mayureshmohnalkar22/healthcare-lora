# Model card

## Intended use

Research and portfolio demonstration of sentence-role classification in published randomized-controlled-trial abstracts. Inputs are single sentences, with five allowable role outputs. No patient-specific inference or clinical recommendation endpoint is provided.

## Models

- Keyword baseline: deterministic hand-selected lexical cues.
- TF-IDF baseline: unigram/bigram vectorizer and logistic regression, exported as non-executable JSON. Class probabilities are uncalibrated.
- LoRA path: Qwen2.5-0.5B-Instruct with PEFT q_proj/v_proj adapters. The base model's own Apache 2.0 terms apply separately from the dataset's source copyright conditions.

## Training and evaluation status

The classical experiment is measured and committed. A real Qwen2.5-0.5B smoke run also trained a rank-8 adapter on 100 public training sentences (540,672 trainable parameters), with a 50-sentence validation subset; its adapter is included in the release ZIP. The tiny-model LoRA integration verifies training, completion masking, checkpointing, adapter export, MLflow logging, adapter reload, and inference with/without the adapter. It does not establish domain performance. Consult `docs/validation.md` for any subsequent real pretrained-model smoke run.

## Failure modes

Background vs objective and results vs conclusion sentences can overlap in meaning. Short or underspecified sentences often need surrounding abstract context. Inputs beyond the configured token budget have their sentence content shortened while preserving the full task instructions and chat delimiters. Truncation is explicitly recorded.

Free generation may produce explanations instead of a role. Strict parsing records those as invalid. Confidence is not invented for generative outputs. The API permits an expected role only to calculate a post-generation match.

## Release policy

The demo baseline is not a substitute for an adapter, and its report cannot pass the LoRA promotion gate. A gate pass does not establish medical safety, generalization to patient notes, or compliance for a healthcare deployment. Concurrency is intentionally limited to one comparison at a time; high-throughput vLLM serving is a follow-up implementation.

The real pretrained-model smoke comparison did not establish an improvement: 20% base vs 24% adapted accuracy on 25 validation sentences, with a −24 to +32 percentage-point interval. The adapter failed the release gate and is bundled only as pipeline evidence.
