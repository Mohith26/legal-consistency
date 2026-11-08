"""Load LegalBench tasks with their official few-shot base prompts."""
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path

# Rule-application and interpretation tasks with short, closed-form answers.
TASKS = {
    "hearsay": None,
    "personal_jurisdiction": None,
    "telemarketing_sales_rule": None,
    "ucc_v_common_law": None,
    "insurance_policy_interpretation": None,
    "abercrombie": None,
    "proa": None,
    "sara_entailment": 100,  # sampled (seed 0) to keep runtime bounded
    "diversity_3": 60,
}

LABELS = {
    "ucc_v_common_law": ["UCC", "Common Law"],
    "insurance_policy_interpretation": ["A", "B", "C"],
    "abercrombie": ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"],
    "sara_entailment": ["Entailment", "Contradiction"],
}
DEFAULT_LABELS = ["Yes", "No"]


@dataclass(frozen=True)
class Item:
    task: str
    index: str
    fields: dict
    answer: str


def labels_for(task):
    return LABELS.get(task, DEFAULT_LABELS)


def load_items(task, data_dir, limit=None):
    with open(Path(data_dir) / f"{task}.tsv", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if limit is not None and limit < len(rows):
        rows = random.Random(0).sample(rows, limit)
    return [Item(task, r["index"], {k: v for k, v in r.items() if k not in ("index", "answer")}, r["answer"].strip())
            for r in rows]


@dataclass
class Prompt:
    preamble: list
    examples: list
    template: str

    @property
    def answer_prefix(self):
        return self.template.splitlines()[-1].strip()

    def render(self, item, order=None):
        examples = self.examples if order is None else [self.examples[i] for i in order]
        query = self.template
        for key, value in item.fields.items():
            query = query.replace("{{" + key + "}}", value)
        return "\n\n".join(self.preamble + examples + [query])


def parse_prompt(text):
    """Split a base prompt into preamble blocks, few-shot example blocks and the query template.

    Example blocks are the blocks whose last line starts with the same answer prefix as the
    template's final line (for example "A:" or "Label:"), so they can be reordered safely.
    """
    blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    template = blocks[-1]
    if "{{" not in template:
        raise ValueError("last block is not a query template")
    prefix = template.splitlines()[-1].strip()
    preamble, examples = [], []
    for block in blocks[:-1]:
        last = block.splitlines()[-1].strip()
        if last.startswith(prefix) and len(last) > len(prefix):
            examples.append(block)
        elif not examples:
            preamble.append(block)
        else:
            # A non-example block after examples started: keep it attached to the previous example.
            examples[-1] = examples[-1] + "\n\n" + block
    return Prompt(preamble, examples, template)


def load_prompt(task, legalbench_dir):
    return parse_prompt((Path(legalbench_dir) / "tasks" / task / "base_prompt.txt").read_text())


def shuffled_orders(n_examples, n_orders, seed=0):
    """Deterministic non-identity permutations of the few-shot examples."""
    rng = random.Random(seed)
    identity = list(range(n_examples))
    orders, seen = [], {tuple(identity)}
    attempts = 0
    while len(orders) < n_orders and attempts < 1000:
        attempts += 1
        perm = identity[:]
        rng.shuffle(perm)
        if tuple(perm) not in seen:
            seen.add(tuple(perm))
            orders.append(perm)
    return orders


def parse_answer(task, response, prefix=None):
    """Map a raw completion to one of the task's labels, or "INVALID".

    Models sometimes echo the answer prefix ("Label: A"); if ``prefix`` is given and the
    completion starts with it, it is stripped first.
    """
    text = response.strip().strip("*\"'`").strip()
    if prefix and text.lower().startswith(prefix.lower()):
        text = text[len(prefix):].strip().strip("*\"'`").strip()
    labels = labels_for(task)
    low = text.lower()
    # Longest label first so "Common Law" wins over a stray "Common".
    for label in sorted(labels, key=len, reverse=True):
        lab = label.lower()
        if low.startswith(lab):
            nxt = low[len(lab):len(lab) + 1]
            if nxt and nxt.isalpha():
                continue  # "Nope", "Answered": not a label
            return label
    return "INVALID"
