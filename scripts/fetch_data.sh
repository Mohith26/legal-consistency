#!/usr/bin/env bash
# Downloads the LegalBench test splits used here and the official few-shot prompts.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
for t in hearsay personal_jurisdiction telemarketing_sales_rule ucc_v_common_law insurance_policy_interpretation \
         abercrombie proa sara_entailment diversity_3; do
  curl -sfL -o "data/$t.tsv" "https://huggingface.co/datasets/nguha/legalbench/resolve/main/data/$t/test.tsv"
done
[ -d legalbench ] || git clone -q --depth 1 https://github.com/HazyResearch/legalbench legalbench
echo "data ready"
