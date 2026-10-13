# Dataset card

## Source and task

PubMed 20k RCT, a subset of PubMed 200k RCT by Franck Dernoncourt and Ji Young Lee (2017). Source: https://github.com/Franck-Dernoncourt/pubmed-rct . Paper: https://aclanthology.org/I17-2052/ .

Use the `PubMed_20k_RCT` variant with numbers retained. Source revision: `17ed2cb0590decfca0266add0c76f254f67232b4`. All download checksums are recorded in `results/dataset-audit.json`.

| Split | Abstracts | Sentences |
|---|---:|---:|
| Train | 15,000 | 180,040 |
| Validation (official dev) | 2,500 | 30,212 |
| Test | 2,500 | 30,135 |

Sentence roles: background, objective, methods, results, conclusions. Source PMID and sentence index are retained. The project predicts **one sentence at a time**; it does not include adjacent sentences or sentence position in its model prompt. That makes comparisons with context-aware sequential classifiers inappropriate without matching their input protocol.

## Integrity policy

Never random-split sentences from a shared abstract. Preserve the official PMID-level partitions. Fail when PMIDs or complete normalized abstracts overlap across splits. Shared normalized sentences are reported and kept to preserve the upstream benchmark: 208 train/dev, 210 train/test, 95 dev/test. Generic boilerplate can recur across papers; the audit does not declare the dataset free of all linguistic overlap.

No synthetic medical corpus is used for training. The small general-retention suite and test fixtures are authored software diagnostics, not medical training data.

Smoke subsets use a deterministic balanced round-robin sample across roles. Their class prevalence differs from the full corpus. For the bundled classical experiment only, training was capped at 20,000 balanced sentences; validation and test used all rows. Validation selects C; the test measures the selected model once. Neither source IDs nor target role labels are placed in the inference text.

## Licensing and distribution

The upstream README does **not** offer a blanket MIT/CC license for all abstract text. It points to NLM's copyright guidance and notes that some abstracts may be copyright-protected. See https://www.nlm.nih.gov/web_policies.html#copyright .

This repository contains download/processing code, provenance, counts, example IDs, and predictions. It does not bundle or relicense the source sentences. `data/raw` and `data/processed` are ignored by git and omitted from the release ZIP. Review upstream terms for your intended use, particularly redistribution.

## Limitations

English RCT literature is not representative of all medical writing or patient records. Labels reflect abstract roles, not factual accuracy. The base LLM may have encountered published benchmark examples during pretraining; this pipeline cannot exclude that. PMID separation does not rule out related trials, near duplicates, or temporal leakage. No real-world clinical validity is established by this benchmark.
