"""The six built-in defences, in scoreboard order. `nikash.scope.DEFENCES`
lists the ones the version 1 study runs."""

from __future__ import annotations

from .base import Defence
from .model_free import RulesVerdict, StructuredExtraction
from .prompt_level import NoDefence, Quarantine, Spotlight, SystemWarning

ORDER = ("none", "warning", "spotlight", "quarantine", "extract", "rules")


def build_defences(ids: list[str] | None = None, *, seed: int = 0, sigma_matcher=None,
                   explain_triage: bool = False) -> list[Defence]:
    factories = {
        "none": NoDefence,
        "warning": SystemWarning,
        "spotlight": lambda: Spotlight(seed=seed),
        "quarantine": Quarantine,
        "extract": StructuredExtraction,
        "rules": lambda: RulesVerdict(sigma_matcher=sigma_matcher, explain_triage=explain_triage),
    }
    wanted = list(ids) if ids else list(ORDER)
    unknown = [i for i in wanted if i not in factories]
    if unknown:
        raise ValueError(f"unknown defence(s): {', '.join(unknown)}")
    return [factories[i]() for i in wanted]


__all__ = ["Defence", "ORDER", "build_defences"]
