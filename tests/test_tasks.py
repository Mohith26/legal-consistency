from pathlib import Path

import pytest

from lc.tasks import TASKS, Item, labels_for, parse_answer, parse_prompt, shuffled_orders

LEGALBENCH = Path(__file__).resolve().parents[1] / "legalbench"

PROMPT = """Hearsay is an out-of-court statement.

Q: First example. Is there hearsay?
A: No

Q: Second example. Is there hearsay?
A: Yes

Q: Third example. Is there hearsay?
A: No

Q: {{text}} Is there hearsay?
A:"""


def test_parse_prompt_splits_preamble_examples_template():
    p = parse_prompt(PROMPT)
    assert p.preamble == ["Hearsay is an out-of-court statement."]
    assert len(p.examples) == 3
    assert p.answer_prefix == "A:"
    item = Item("hearsay", "0", {"text": "Query."}, "No")
    rendered = p.render(item)
    assert rendered.endswith("Q: Query. Is there hearsay?\nA:")
    assert "{{" not in rendered


def test_render_with_order_reorders_examples_only():
    p = parse_prompt(PROMPT)
    item = Item("hearsay", "0", {"text": "Query."}, "No")
    r = p.render(item, order=[2, 0, 1])
    assert r.index("Third example") < r.index("First example") < r.index("Second example")
    assert r.startswith("Hearsay is an out-of-court statement.")
    assert r.endswith("A:")


def test_shuffled_orders_are_distinct_non_identity_permutations():
    orders = shuffled_orders(5, 3)
    assert len(orders) == 3
    assert len({tuple(o) for o in orders}) == 3
    for o in orders:
        assert sorted(o) == list(range(5))
        assert o != list(range(5))
    assert shuffled_orders(5, 3) == orders  # deterministic


def test_shuffled_orders_small_n():
    assert shuffled_orders(1, 2) == []  # nothing to reorder
    assert shuffled_orders(2, 5) == [[1, 0]]


@pytest.mark.parametrize("task,response,prefix,expected", [
    ("hearsay", "No\n\nExplanation", "A:", "No"),
    ("hearsay", "A: Yes", "A:", "Yes"),
    ("hearsay", "Nope", "A:", "INVALID"),
    ("ucc_v_common_law", "Governed by: Common Law", "Governed by:", "Common Law"),
    ("ucc_v_common_law", "UCC.", "Governed by:", "UCC"),
    ("insurance_policy_interpretation", "Label: A\n", "Label:", "A"),
    ("insurance_policy_interpretation", "A: Yes", "Label:", "A"),
    ("insurance_policy_interpretation", "Ambiguous", "Label:", "INVALID"),
    ("sara_entailment", "Answer: Contradiction", "Answer:", "Contradiction"),
    ("abercrombie", "Suggestive", "A:", "suggestive"),
])
def test_parse_answer(task, response, prefix, expected):
    assert parse_answer(task, response, prefix) == expected


def test_labels_default_yes_no():
    assert labels_for("hearsay") == ["Yes", "No"]


@pytest.mark.skipif(not LEGALBENCH.exists(), reason="legalbench checkout not present")
@pytest.mark.parametrize("task", list(TASKS))
def test_official_prompts_parse(task):
    p = parse_prompt((LEGALBENCH / "tasks" / task / "base_prompt.txt").read_text())
    assert len(p.examples) >= 3
    assert "{{" in p.template
    for ex in p.examples:
        assert ex.splitlines()[-1].startswith(p.answer_prefix)
