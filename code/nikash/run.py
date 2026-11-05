"""Run every (model, defence) pair over a set of cases, then judge the answers
if a judge is given."""

from __future__ import annotations

from typing import Callable

from .defences import Defence
from .schema import Case

MAX_TEXT_KEPT = 8000   # a whole answer, even extract's 1,024 tokens, so it can be re-read later


def _record(case: Case, defence_id: str, model_name: str, response, error, text: str) -> dict:
    return {
        "case_id": case.id, "benchmark": case.benchmark, "task": case.task,
        "defence": defence_id, "model": model_name,
        "ok": response is not None, "error": error, "response": response,
        "model_text": text[:MAX_TEXT_KEPT],
    }


def run_matrix(cases: list[Case], defences: list[Defence], models: list, judge=None,
               log: Callable[[str], None] = lambda _msg: None) -> list[dict]:
    """With `judge=None` (version 1) nothing is judged. Summaries can't be
    scored without a judge, so summary cases are then refused."""
    if judge is None and any(c.task == "summary" for c in cases):
        raise ValueError("summary cases need a judge")
    records: list[dict] = []
    for model in models:
        for defence in defences:
            for task in ("triage", "summary"):
                subset = [c for c in cases if c.task == task]
                if not subset:
                    continue
                asked = [c for c in subset if defence.needs_model(c)]
                log(f"{model.name} · {defence.id} · {task}: {len(subset)} cases, {len(asked)} model calls")
                limit = {"max_new_tokens": defence.max_new_tokens} if getattr(defence, "max_new_tokens", None) else {}
                answers = iter(model.generate_many([(defence.prepare(c), c) for c in asked], **limit) if asked else [])
                for case in subset:
                    text = next(answers) if defence.needs_model(case) else ""
                    response, error = defence.decide(case, text)
                    records.append(_record(case, defence.id, model.name, response, error, text))
    if judge is not None:
        attach_judgements(cases, records, judge, {d.id: d.model_decides for d in defences})
    return records


def attach_judgements(cases: list[Case], records: list[dict], judge, model_decides: dict[str, bool]) -> None:
    """Summaries get fact judgements; triage answers from defences where the
    model decides get an explanation-stance judgement."""
    by_id = {c.id: c for c in cases}
    todo = [
        r for r in records
        if r["ok"] and (r["task"] == "summary" or model_decides.get(r["defence"], True))
    ]
    results = judge.judge_many([(by_id[r["case_id"]], r["response"]) for r in todo])
    for r, j in zip(todo, results):
        r["judge"] = j
