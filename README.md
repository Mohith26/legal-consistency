# legal-consistency

Norm Ai's Legal AGI Lab reported that frontier models reach the same conclusion on a legal question only about 90% of the time. I wanted to know two things that number doesn't say: does setting temperature to 0 actually fix it, and is the disagreement random or does it land on questions that are genuinely unclear?

I ran the same kind of measurement on two open-weight models locally (Ollama on an Apple Silicon laptop, no paid APIs) over LegalBench rule-application and interpretation tasks, using LegalBench's own few-shot prompts.

## Setup

- **Questions:** 513 LegalBench test questions from six tasks run on both models (hearsay, personal_jurisdiction, telemarketing_sales_rule, ucc_v_common_law, insurance_policy_interpretation, proa). Qwen also ran abercrombie, a 100-question sample of sara_entailment and a 60-question sample of diversity_3 (per-task table only).
- **Models:** `qwen2.5:7b` and `llama3.1:8b` (Q4_K_M, Ollama).
- **Per question, 11 calls:**
  - 8 samples at temperature 0.7 (seeds 0-7), official few-shot prompt;
  - 1 greedy answer (temperature 0);
  - 2 greedy answers with the few-shot examples **reordered** (same examples, same wording, different order).
- **Metrics:** pairwise agreement (chance two sampled runs give the same answer, the "same conclusion X% of the time" number), 8/8 unanimity, the share of greedy answers that change under reordering, and accuracy.
- 15,136 model calls in the main runs (plus 6,688 in the discarded llama chat-mode run). Raw outputs for every call are in `results/`.

## Results

| model | questions | pairwise agreement (T=0.7) | 8/8 unanimous | greedy answer flips when examples are reordered | flips among 8/8-unanimous questions | accuracy (greedy) | accuracy when 8/8 unanimous | accuracy when split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5:7b | 513 | 93.3% | 85.0% | 19.3% | 11.9% | 73.5% | 77.8% (85% of questions) | 45.5% |
| llama3.1:8b | 513 | 66.3% | 34.5% | 12.5% | 1.1% | 69.8% | 94.9% (35% of questions) | 56.2% |

**Insurance policy interpretation**, where LegalBench's lawyer labels mark 59 claims as "ambiguous" and 74 as clearly covered or not:

| model | metric | ambiguous | clear | Fisher exact p |
|---|---|---:|---:|---:|
| qwen2.5:7b | greedy flips under reordering | 55.9% | 31.1% | 0.005 |
| qwen2.5:7b | not 8/8 unanimous | 28.8% | 17.6% | 0.146 |
| llama3.1:8b | greedy flips under reordering | 16.9% | 12.2% | 0.464 |
| llama3.1:8b | not 8/8 unanimous | 98.3% | 86.5% | 0.023 |

Full per-task tables: [`results/RESULTS.md`](results/RESULTS.md).

## What I take from it

1. **The ~90% number reproduces, but it measures only one kind of inconsistency.** Qwen agrees with itself 93% of the time under sampling, close to the Lab's frontier-model figure. But one in five of its temperature-0 conclusions changes when nothing changes except the order of the worked examples, and 12% of questions where it was 8-for-8 unanimous still flip. Temperature 0 makes a model repeatable, not consistent.
2. **The two kinds of instability don't track each other.** Llama is the mirror image: much noisier under sampling (66%) but more stable to reordering (12.5%). At similar accuracy, a single consistency number would rank these models the opposite way depending on which test you run.
3. **Instability concentrates on genuinely ambiguous law.** On claims lawyers labeled ambiguous, each model gets significantly less stable on one of the two axes (qwen: reorder flips 56% vs 31%, p = 0.005; llama: split votes 98% vs 87%, p = 0.023). That makes disagreement a useful signal for routing to a human rather than pure noise.
4. **Agreement is a usable abstention rule, with a model-specific trade-off.** Answering only when all 8 samples agree gives llama 94.9% accuracy on 35% of questions and qwen 77.8% on 85%.

## Limitations (please read)

- These are 7-8B open models, not the frontier models in Norm's post, and the Lab's exact protocol isn't public in detail, so this is the same kind of measurement, not a replication of their numbers.
- **Prompt mode differs between the models.** Qwen was run through Ollama's chat template and answers in the few-shot format. Llama under the chat template ignored the format 97% of the time (it writes essays such as "Based on the definitions..."), so I reran it as plain text completion, which is how LegalBench prompts are meant to be used. The chat-mode run is kept in `results/raw_llama_chat.jsonl`. Cross-model comparisons are therefore confounded by prompt mode; the within-model findings are the claims I'd stand behind.
- Answers are parsed from the first 8 generated tokens. Qwen often restated the question on abercrombie (49% unparseable) and some sara_entailment items (6%), so those tasks are excluded from the pooled table.
- Two reorderings per question is a small sample of the 24 to 720 possible orders, so the reorder flip rates are lower bounds.
- Temperature 0.7 is my choice; the Lab's sampling settings may differ.

## Reproduce

```bash
ollama pull qwen2.5:7b && ollama pull llama3.1:8b
./scripts/fetch_data.sh                       # LegalBench test splits + official prompts
python -m lc.run --model qwen2.5:7b --out results/raw_qwen.jsonl
python -m lc.run --model llama3.1:8b --raw-prompt --out results/raw_llama.jsonl \
  --tasks hearsay personal_jurisdiction telemarketing_sales_rule ucc_v_common_law insurance_policy_interpretation abercrombie proa
python -m lc.analyze                           # writes results/RESULTS.md and results/summary.json
python -m pytest                               # 32 tests (parsing, prompt reordering, metrics, Fisher test, aggregation)
```

The runner is resumable (it skips calls already in the output file). Pure standard library; no Python dependencies beyond pytest for tests.

LegalBench: Guha et al., NeurIPS 2023 (https://github.com/HazyResearch/legalbench); each task keeps its own license. Data is downloaded, not redistributed here.
