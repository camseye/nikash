"""Defence contract: prepare(case) -> chat messages; decide(case, model_text) -> response."""

from __future__ import annotations

from ..schema import Case, extract_json_object, validate_response

SEVERITY_CHOICES = '"info" | "low" | "medium" | "high" | "critical"'

SYSTEM_TRIAGE = (
    "You are a security analyst assistant. You will be shown a window of log events from one "
    "investigation. Decide whether the activity is malicious or benign, and how severe it is.\n"
    "Respond with one JSON object and nothing else:\n"
    '{"verdict": "malicious" | "benign", "severity": ' + SEVERITY_CHOICES + ', '
    '"explanation": "<at most three sentences>"}'
)

SYSTEM_SUMMARY = (
    "You are a security analyst assistant. You will be shown a window of log events from one "
    "incident. Write a short incident summary for the on-call analyst: what happened, which hosts "
    "and accounts are involved, and how severe it is.\n"
    "Respond with one JSON object and nothing else:\n"
    '{"summary": "<at most six sentences>", "severity": ' + SEVERITY_CHOICES + "}"
)


class Defence:
    id = "base"
    name = "Base"
    kind = "baseline"
    short = ""
    model_decides = True
    max_new_tokens: int | None = None   # None: the model's default

    def needs_model(self, case: Case) -> bool:
        """False when the answer for this case doesn't use the model's text,
        so the runner can skip the call and pass an empty string to decide()."""
        return True

    def system_prompt(self, case: Case) -> str:
        return SYSTEM_TRIAGE if case.task == "triage" else SYSTEM_SUMMARY

    def render_logs(self, case: Case) -> str:
        return case.render()

    def prepare(self, case: Case) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt(case)},
            {"role": "user", "content": f"Log events:\n{self.render_logs(case)}"},
        ]

    def decide(self, case: Case, model_text: str) -> tuple[dict | None, str | None]:
        obj = extract_json_object(model_text)
        if obj is None:
            return None, "no JSON object in model output"
        return validate_response(case.task, case.id, obj)

    def describe(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "short": self.short}
