import math

import pytest

from lc.metrics import contradiction_rate_at_scale, majority, pairwise_agreement, unanimous


def test_pairwise_agreement_unanimous_is_one():
    assert pairwise_agreement(["Yes"] * 8) == 1.0


def test_pairwise_agreement_split():
    # 7 Yes / 1 No: pairs that agree = C(7,2)=21 of C(8,2)=28
    assert math.isclose(pairwise_agreement(["Yes"] * 7 + ["No"]), 21 / 28)
    # 4/4 split: 2*C(4,2)=12 of 28
    assert math.isclose(pairwise_agreement(["Yes"] * 4 + ["No"] * 4), 12 / 28)


def test_pairwise_agreement_needs_two():
    with pytest.raises(ValueError):
        pairwise_agreement(["Yes"])


def test_unanimous():
    assert unanimous(["A", "A"])
    assert not unanimous(["A", "B"])


def test_majority_tie_break_prefers_greedy():
    assert majority(["Yes", "No"], tie_break="No") == "No"
    assert majority(["Yes", "No"]) == "No"  # lexicographic fallback
    assert majority(["Yes", "Yes", "No"], tie_break="No") == "Yes"


def test_contradiction_rate_at_scale():
    # 90% agreement, question asked 10 times: 45 pairs, 4.5 expected contradictory pairs
    assert math.isclose(contradiction_rate_at_scale(0.9, 10), 4.5)


def test_fisher_exact_matches_known_values():
    from lc.metrics import fisher_exact_two_sided
    # Classic tea-tasting table: p = 0.4857 (two-sided)
    assert math.isclose(fisher_exact_two_sided(3, 1, 1, 3), 0.4857142857, rel_tol=1e-6)
    # scipy.stats.fisher_exact([[8, 2], [1, 5]]) -> p = 0.03496503
    assert math.isclose(fisher_exact_two_sided(8, 2, 1, 5), 0.0349650349, rel_tol=1e-6)
    assert fisher_exact_two_sided(5, 5, 5, 5) == 1.0
