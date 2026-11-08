# Results

## Overall (core tasks pooled: hearsay, personal_jurisdiction, telemarketing_sales_rule, ucc_v_common_law, insurance_policy_interpretation, proa)

| model | questions | pairwise agreement | unanimous (8/8) | greedy flips under few-shot reorder | flips among unanimous | acc greedy | acc single sample | acc majority-of-8 | acc when unanimous | acc when split | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3.1:8b | 513 | 66.3% | 34.5% | 12.5% | 1.1% | 69.8% | 66.0% | 69.6% | 94.9% | 56.2% | 0.0% |
| qwen2.5:7b | 513 | 93.3% | 85.0% | 19.3% | 11.9% | 73.5% | 73.1% | 72.9% | 77.8% | 45.5% | 0.1% |

## Insurance policy interpretation: clauses lawyers labeled ambiguous vs clear

| model | ambiguous n | agreement | unanimous | greedy flips | clear n | agreement | unanimous | greedy flips | Fisher p (flips) | Fisher p (split) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3.1:8b | 59 | 42.6% | 1.7% | 16.9% | 74 | 52.8% | 13.5% | 12.2% | 0.464 | 0.023 |
| qwen2.5:7b | 59 | 85.6% | 71.2% | 55.9% | 74 | 91.2% | 82.4% | 31.1% | 0.005 | 0.146 |

## Per task

| model | task | n | agreement | unanimous | greedy flips | acc greedy | acc majority | invalid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama3.1:8b | abercrombie | 95 | 54.7% | 21.1% | 35.8% | 52.6% | 52.6% | 0.2% |
| llama3.1:8b | hearsay | 94 | 58.0% | 16.0% | 23.4% | 74.5% | 74.5% | 0.0% |
| llama3.1:8b | insurance_policy_interpretation | 133 | 48.3% | 8.3% | 14.3% | 42.1% | 41.4% | 0.0% |
| llama3.1:8b | personal_jurisdiction | 50 | 61.6% | 20.0% | 16.0% | 54.0% | 54.0% | 0.0% |
| llama3.1:8b | proa | 95 | 82.4% | 60.0% | 5.3% | 95.8% | 95.8% | 0.0% |
| llama3.1:8b | telemarketing_sales_rule | 47 | 58.6% | 17.0% | 4.3% | 70.2% | 70.2% | 0.0% |
| llama3.1:8b | ucc_v_common_law | 94 | 90.3% | 80.9% | 8.5% | 86.2% | 86.2% | 0.0% |
| qwen2.5:7b | abercrombie | 95 | 67.1% | 32.6% | 37.9% | 35.8% | 35.8% | 49.2% |
| qwen2.5:7b | diversity_3 | 60 | 97.3% | 95.0% | 15.0% | 61.7% | 61.7% | 0.0% |
| qwen2.5:7b | hearsay | 94 | 91.3% | 80.9% | 20.2% | 72.3% | 72.3% | 0.0% |
| qwen2.5:7b | insurance_policy_interpretation | 133 | 88.7% | 77.4% | 42.1% | 50.4% | 48.1% | 0.0% |
| qwen2.5:7b | personal_jurisdiction | 50 | 92.6% | 84.0% | 18.0% | 64.0% | 64.0% | 0.0% |
| qwen2.5:7b | proa | 95 | 98.8% | 97.9% | 4.2% | 94.7% | 94.7% | 0.0% |
| qwen2.5:7b | sara_entailment | 100 | 82.1% | 66.0% | 21.0% | 66.0% | 69.0% | 6.4% |
| qwen2.5:7b | telemarketing_sales_rule | 47 | 95.6% | 89.4% | 6.4% | 74.5% | 74.5% | 0.0% |
| qwen2.5:7b | ucc_v_common_law | 94 | 95.7% | 85.1% | 8.5% | 90.4% | 90.4% | 0.4% |
