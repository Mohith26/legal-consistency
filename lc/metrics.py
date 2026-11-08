"""Consistency metrics over repeated runs of the same question."""
from collections import Counter


def pairwise_agreement(answers):
    """Probability that two distinct runs (drawn without replacement) give the same answer."""
    k = len(answers)
    if k < 2:
        raise ValueError("need at least two answers")
    counts = Counter(answers)
    return sum(c * (c - 1) for c in counts.values()) / (k * (k - 1))


def unanimous(answers):
    return len(set(answers)) == 1


def majority(answers, tie_break=None):
    """Most common answer. Ties go to ``tie_break`` if it is among the tied answers, else the
    lexicographically smallest tied answer, so the result is deterministic."""
    counts = Counter(answers)
    top = max(counts.values())
    tied = sorted(a for a, c in counts.items() if c == top)
    if tie_break in tied:
        return tie_break
    return tied[0]


def contradiction_rate_at_scale(agreement, repeats_per_week):
    """Expected number of contradictory answer pairs per week if the same question is asked
    ``repeats_per_week`` times, given a pairwise agreement rate."""
    pairs = repeats_per_week * (repeats_per_week - 1) / 2
    return pairs * (1 - agreement)


def fisher_exact_two_sided(a, b, c, d):
    """Two-sided Fisher exact test p-value for the 2x2 table [[a, b], [c, d]]."""
    from math import comb

    n1, n2, k = a + b, c + d, a + c
    total = comb(n1 + n2, k)

    def prob(x):
        return comb(n1, x) * comb(n2, k - x) / total

    observed = prob(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(p for p in (prob(x) for x in range(lo, hi + 1)) if p <= observed * (1 + 1e-9)))
