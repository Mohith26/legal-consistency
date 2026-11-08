import json

from lc.analyze import build, to_markdown


def _rec(model, task, idx, cond, pred, gold):
    return {"model": model, "task": task, "index": idx, "condition": cond, "pred": pred, "gold": gold}


def test_build_end_to_end(tmp_path):
    rows = []
    # q1: unanimous and correct, stable under reorder
    rows += [_rec("m", "hearsay", "1", "greedy", "Yes", "Yes")]
    rows += [_rec("m", "hearsay", "1", f"sample_{i}", "Yes", "Yes") for i in range(4)]
    rows += [_rec("m", "hearsay", "1", "shuffle_0", "Yes", "Yes")]
    # q2: split 2/2, greedy flips under reorder, majority wrong via greedy tie-break
    rows += [_rec("m", "hearsay", "2", "greedy", "No", "Yes")]
    rows += [_rec("m", "hearsay", "2", f"sample_{i}", p, "Yes") for i, p in enumerate(["Yes", "No", "No", "Yes"])]
    rows += [_rec("m", "hearsay", "2", "shuffle_0", "Yes", "Yes")]
    # q3: incomplete (no shuffle) -> excluded
    rows += [_rec("m", "hearsay", "3", "greedy", "No", "No")]
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    s = build(raw, core_tasks=None)["by_model"]["m"]
    assert s["n"] == 2
    assert s["unanimous"] == 0.5
    assert abs(s["agreement"] - (1.0 + 2 / 6) / 2) < 1e-9
    assert s["greedy_flip_rate"] == 0.5
    assert s["acc_greedy"] == 0.5
    assert s["acc_majority"] == 0.5
    assert s["acc_when_unanimous"] == 1.0
    assert s["acc_when_split"] == 0.0
    assert s["hidden_instability"] == 0.0  # the unanimous question did not flip under reorder
    assert "| m | 2 |" in to_markdown(build(raw, core_tasks=None))
    assert build(raw)["by_model"]["m"]["n"] == 2  # hearsay is a core task
