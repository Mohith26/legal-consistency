"""Run the consistency experiment. Resumable: results are appended to a JSONL cache.

Conditions per question:
  greedy      temperature 0, canonical few-shot order
  sample_i    temperature 0.7, seed i, canonical order (i = 0..k-1)
  shuffle_j   temperature 0, few-shot examples reordered by permutation j
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import ollama
from .tasks import TASKS, load_items, load_prompt, parse_answer, shuffled_orders


def jobs_for(model, k, n_shuffles, data_dir, legalbench_dir, tasks=None):
    for task, limit in TASKS.items():
        if tasks and task not in tasks:
            continue
        prompt = load_prompt(task, legalbench_dir)
        orders = shuffled_orders(len(prompt.examples), n_shuffles)
        for item in load_items(task, data_dir, limit):
            yield (model, task, item, "greedy", 0.0, 0, prompt.render(item), prompt.answer_prefix)
            for i in range(k):
                yield (model, task, item, f"sample_{i}", 0.7, i, prompt.render(item), prompt.answer_prefix)
            for j, order in enumerate(orders):
                yield (model, task, item, f"shuffle_{j}", 0.0, 0, prompt.render(item, order), prompt.answer_prefix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--shuffles", type=int, default=2)
    ap.add_argument("--data", default="data")
    ap.add_argument("--legalbench", default="legalbench")
    ap.add_argument("--out", default="results/raw.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tasks", nargs="*", help="subset of task names (default: all)")
    ap.add_argument("--raw-prompt", action="store_true", help="plain completion, no chat template")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as f:
            for line in f:
                r = json.loads(line)
                done.add((r["model"], r["task"], r["index"], r["condition"]))

    todo = [j for j in jobs_for(args.model, args.k, args.shuffles, args.data, args.legalbench, args.tasks)
            if (j[0], j[1], j[2].index, j[3]) not in done]
    print(f"{args.model}: {len(todo)} calls to run ({len(done)} cached)", flush=True)

    def call(job):
        model, task, item, cond, temp, seed, prompt, prefix = job
        resp = ollama.generate(model, prompt, temp, seed, raw=args.raw_prompt)
        return {"model": model, "task": task, "index": item.index, "condition": cond, "temperature": temp,
                "seed": seed, "prompt_mode": "raw" if args.raw_prompt else "chat", "gold": item.answer, "response": resp, "pred": parse_answer(task, resp, prefix)}

    with ThreadPoolExecutor(args.workers) as ex, out.open("a") as f:
        for n, rec in enumerate(ex.map(call, todo), 1):
            f.write(json.dumps(rec) + "\n")
            if n % 500 == 0:
                f.flush()
                print(f"  {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
