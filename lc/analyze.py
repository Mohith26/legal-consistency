"""Aggregate raw runs into the results tables (results/summary.json + results/RESULTS.md)."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from .metrics import fisher_exact_two_sided, majority, pairwise_agreement, unanimous


def load(path):
    by_q = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_q[(r["model"], r["task"], r["index"])][r["condition"]] = r
    return by_q


def question_stats(conds):
    samples = [conds[c]["pred"] for c in sorted(conds) if c.startswith("sample_")]
    shuffles = [conds[c]["pred"] for c in sorted(conds) if c.startswith("shuffle_")]
    greedy = conds["greedy"]["pred"]
    gold = conds["greedy"]["gold"]
    return {
        "gold": gold,
        "greedy": greedy,
        "samples": samples,
        "agreement": pairwise_agreement(samples),
        "unanimous": unanimous(samples),
        "majority": majority(samples, tie_break=greedy),
        "greedy_flip": any(s != greedy for s in shuffles),
        "invalid": sum(p == "INVALID" for p in samples + shuffles + [greedy]),
        "n_calls": len(samples) + len(shuffles) + 1,
    }


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(stats):
    n = len(stats)
    return {
        "n": n,
        "agreement": mean(s["agreement"] for s in stats),
        "unanimous": mean(s["unanimous"] for s in stats),
        "acc_greedy": mean(s["greedy"] == s["gold"] for s in stats),
        "acc_single_sample": mean(mean(p == s["gold"] for p in s["samples"]) for s in stats),
        "acc_majority": mean(s["majority"] == s["gold"] for s in stats),
        "greedy_flip_rate": mean(s["greedy_flip"] for s in stats),
        "invalid_rate": sum(s["invalid"] for s in stats) / max(1, sum(s["n_calls"] for s in stats)),
        "coverage_unanimous": mean(s["unanimous"] for s in stats),
        "hidden_instability": mean(s["greedy_flip"] for s in stats if s["unanimous"]),
        "acc_when_unanimous": mean(s["majority"] == s["gold"] for s in stats if s["unanimous"]),
        "acc_when_split": mean(s["majority"] == s["gold"] for s in stats if not s["unanimous"]),
    }


# Tasks pooled in the headline table: run on both models and parsed cleanly (<2% invalid answers).
# abercrombie (qwen often restates the question instead of answering within 8 tokens) and the
# qwen-only sara_entailment / diversity_3 runs are reported per task only.
CORE_TASKS = ("hearsay", "personal_jurisdiction", "telemarketing_sales_rule", "ucc_v_common_law",
              "insurance_policy_interpretation", "proa")


def build(raw_paths, complete_only=True, core_tasks=CORE_TASKS):
    if isinstance(raw_paths, (str, bytes)) or hasattr(raw_paths, "__fspath__"):
        raw_paths = [raw_paths]
    by_q = {}
    for path in raw_paths:
        by_q.update(load(path))
    per = defaultdict(list)
    for (model, task, _idx), conds in by_q.items():
        if complete_only and ("greedy" not in conds or sum(c.startswith("sample_") for c in conds) < 2
                              or not any(c.startswith("shuffle_") for c in conds)):
            continue
        per[(model, task)].append(question_stats(conds))
    out = {"by_model_task": {}, "by_model": {}, "insurance_ambiguity": {}}
    models = sorted({m for m, _ in per})
    for (model, task), stats in sorted(per.items()):
        out["by_model_task"][f"{model}|{task}"] = summarize(stats)
    for model in models:
        all_stats = [s for (m, t), st in per.items() if m == model and (not core_tasks or t in core_tasks) for s in st]
        out["by_model"][model] = summarize(all_stats)
        ins = per.get((model, "insurance_policy_interpretation"), [])
        amb = [s for s in ins if s["gold"] == "C"]
        clear = [s for s in ins if s["gold"] in ("A", "B")]
        if ins:
            out["insurance_ambiguity"][model] = {
                "ambiguous_n": len(amb), "ambiguous_agreement": mean(s["agreement"] for s in amb),
                "ambiguous_unanimous": mean(s["unanimous"] for s in amb),
                "ambiguous_greedy_flip": mean(s["greedy_flip"] for s in amb),
                "clear_n": len(clear), "clear_agreement": mean(s["agreement"] for s in clear),
                "clear_unanimous": mean(s["unanimous"] for s in clear),
                "clear_greedy_flip": mean(s["greedy_flip"] for s in clear),
                "greedy_flip_fisher_p": fisher_exact_two_sided(
                    sum(s["greedy_flip"] for s in amb), sum(not s["greedy_flip"] for s in amb),
                    sum(s["greedy_flip"] for s in clear), sum(not s["greedy_flip"] for s in clear)),
                "split_fisher_p": fisher_exact_two_sided(
                    sum(not s["unanimous"] for s in amb), sum(s["unanimous"] for s in amb),
                    sum(not s["unanimous"] for s in clear), sum(s["unanimous"] for s in clear)),
            }
    return out


def pct(x):
    return f"{100 * x:.1f}%"


def to_markdown(summary):
    lines = ["# Results", "", "## Overall (core tasks pooled: " + ", ".join(CORE_TASKS) + ")", "",
             "| model | questions | pairwise agreement | unanimous (8/8) | greedy flips under few-shot reorder | flips among unanimous | acc greedy | acc single sample | acc majority-of-8 | acc when unanimous | acc when split | invalid |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m, s in summary["by_model"].items():
        lines.append(f"| {m} | {s['n']} | {pct(s['agreement'])} | {pct(s['unanimous'])} | {pct(s['greedy_flip_rate'])} | "
                     f"{pct(s['hidden_instability'])} | {pct(s['acc_greedy'])} | {pct(s['acc_single_sample'])} | {pct(s['acc_majority'])} | "
                     f"{pct(s['acc_when_unanimous'])} | {pct(s['acc_when_split'])} | {pct(s['invalid_rate'])} |")
    lines += ["", "## Insurance policy interpretation: clauses lawyers labeled ambiguous vs clear", "",
              "| model | ambiguous n | agreement | unanimous | greedy flips | clear n | agreement | unanimous | greedy flips | Fisher p (flips) | Fisher p (split) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m, s in summary["insurance_ambiguity"].items():
        lines.append(f"| {m} | {s['ambiguous_n']} | {pct(s['ambiguous_agreement'])} | {pct(s['ambiguous_unanimous'])} | "
                     f"{pct(s['ambiguous_greedy_flip'])} | {s['clear_n']} | {pct(s['clear_agreement'])} | "
                     f"{pct(s['clear_unanimous'])} | {pct(s['clear_greedy_flip'])} | {s['greedy_flip_fisher_p']:.3f} | "
                     f"{s['split_fisher_p']:.3f} |")
    lines += ["", "## Per task", "",
              "| model | task | n | agreement | unanimous | greedy flips | acc greedy | acc majority | invalid |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, s in summary["by_model_task"].items():
        m, t = key.split("|")
        lines.append(f"| {m} | {t} | {s['n']} | {pct(s['agreement'])} | {pct(s['unanimous'])} | {pct(s['greedy_flip_rate'])} | "
                     f"{pct(s['acc_greedy'])} | {pct(s['acc_majority'])} | {pct(s['invalid_rate'])} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", default=["results/raw_qwen.jsonl", "results/raw_llama.jsonl"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    summary = build(args.raw)
    Path(args.out, "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    Path(args.out, "RESULTS.md").write_text(to_markdown(summary))
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
