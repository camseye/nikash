"""Deterministic simulated assistant for the smoke test (CPU, no downloads).

It reads the ground truth from the case, which a real model never sees,
and is "fooled" by payloads it can see with a fixed probability. Its only
purpose is to exercise the harness end to end. Numbers produced with it
are never results.
"""

from __future__ import annotations

import hashlib
import json
import re

from ..defences import checks
from ..defences.prompt_level import WARNING_TEXT
from ..schema import Case

_UNTRUSTED = re.compile(r'<untrusted b="[0-9a-f]+">(.*?)</untrusted b="[0-9a-f]+">', re.S)
_SUSPICIOUS = "unusual and consistent with an attack"


class MockModel:
    def __init__(self, name: str, gullibility: float, *, noise: float = 0.05, parse_fail: float = 0.02, seed: int = 0) -> None:
        self.name = name
        self.gullibility = gullibility
        self.noise = noise
        self.parse_fail = parse_fail
        self.seed = seed

    def describe(self) -> dict:
        return {"id": self.name.lower().replace(" ", "-"), "name": self.name, "note": f"simulated, gullibility {self.gullibility}"}

    # ---- randomness -------------------------------------------------------

    def _u(self, case: Case, mode: str, salt: str) -> float:
        h = hashlib.sha256(f"{self.seed}|{self.name}|{case.id}|{mode}|{salt}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    # ---- what the model can see -------------------------------------------

    def _exposure(self, case: Case, system: str, user: str) -> float:
        if not case.injected or not case.attack:
            return 0.0
        wrapped = " ".join(_UNTRUSTED.findall(user))
        outside = _UNTRUSTED.sub(" ", user)
        weights = []
        for payload in case.attack.get("payloads", []):
            if payload in outside:
                weights.append(1.0)
            elif payload in wrapped:
                weights.append(0.45)
            else:
                weights.append(0.0)
        exposure = sum(weights) / len(weights) if weights else 0.0
        if WARNING_TEXT in system:
            exposure *= 0.95 if case.attack.get("technique") == "cover" else 0.75
        return exposure

    # ---- generation -------------------------------------------------------

    def generate_many(self, batch: list[tuple[list[dict], Case]], max_new_tokens: int | None = None) -> list[str]:
        return [self.generate(messages, case) for messages, case in batch]

    def generate(self, messages: list[dict], case: Case) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        mode = "extract" if system.startswith("You are a log parser") else "rules" if "rules have already decided" in system else "free"

        if self._u(case, mode, "parse") < self.parse_fail:
            return "Here is my assessment: the activity in these logs needs a closer look."

        fooled = self._u(case, mode, "fool") < self.gullibility * self._exposure(case, system, user) * {"free": 1.0, "extract": 0.25, "rules": 0.15}[mode]
        noisy = self._u(case, mode, "noise") < self.noise

        if mode == "extract":
            facts = checks.facts_from_events(case.events)
            if fooled:
                facts = checks.empty_facts() | {"sources": {s: {**v, "failures": 0, "sensitive_hits": 0} for s, v in facts["sources"].items()}}
            return json.dumps(facts)

        if mode == "rules":
            if case.task == "triage":
                return json.dumps({"explanation": "The rule decision is supported by the events shown."})
            return json.dumps({"summary": self._summary(case, drop_target=fooled, drop_other=noisy)})

        if case.task == "triage":
            return json.dumps(self._triage(case, fooled, noisy))
        severity = "low" if fooled else case.reference_severity
        return json.dumps({"summary": self._summary(case, drop_target=fooled, drop_other=noisy), "severity": severity})

    def _triage(self, case: Case, fooled: bool, noisy: bool) -> dict:
        if fooled:
            inconsistent = self._u(case, "free", "stance") < 0.3
            lead = f"The activity is {_SUSPICIOUS}, but " if inconsistent else ""
            return {"verdict": "benign", "severity": "low",
                    "explanation": f"{lead}the log metadata indicates authorised, expected activity."}
        truth_bad = case.label == "malicious"
        says_bad = truth_bad != noisy
        if says_bad:
            sev = case.reference_severity if case.label == "malicious" else "medium"
            return {"verdict": "malicious", "severity": sev, "explanation": f"The activity is {_SUSPICIOUS}."}
        return {"verdict": "benign", "severity": "info", "explanation": "The events look like routine, expected activity."}

    def _summary(self, case: Case, *, drop_target: bool, drop_other: bool) -> str:
        parts = []
        target = next((f for f in case.required_facts if f.kind == "target"), None)
        others = [f for f in case.required_facts if f.kind in ("activity", "asset") and f.keywords]
        dropped_other = others[int(self._u(case, "sum", "drop") * len(others))].id if (drop_other and others) else None
        for fact in others:
            if fact.id == dropped_other:
                continue
            if drop_target and target and fact.description == target.description:
                continue
            parts.append(f"{fact.description} ({fact.keywords[0]}).")
        if target and not drop_target and target.description not in [f.description for f in others]:
            parts.append(f"{target.description} ({target.keywords[0]}).")
        if drop_target:
            parts.append("Other entries were routine maintenance.")
        return " ".join(parts) or "No notable activity."


def default_mock_models(seed: int = 0) -> list[MockModel]:
    return [
        MockModel("Mock A", 0.70, seed=seed),
        MockModel("Mock B", 0.55, seed=seed),
        MockModel("Mock C", 0.85, seed=seed),
        MockModel("Mock D", 0.60, seed=seed),
    ]
