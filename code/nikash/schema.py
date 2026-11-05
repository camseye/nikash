"""Common log schema and the nikash/0 plug-in protocol (G2 design §1–§2)."""

from __future__ import annotations

import copy
import functools
import json
import re
from dataclasses import asdict, dataclass, field
from json.decoder import scanstring
from typing import Callable

from . import PROTOCOL

VERDICTS = ("malicious", "benign")
SEVERITIES = ("info", "low", "medium", "high", "critical")
TASKS = ("triage", "summary")

MAX_EXPLANATION = 600
MAX_SUMMARY = 1200

# `{{}}` (no name) stands for a literal "{", so text copied from a log line
# never reads as a placeholder; see `literal`.
_PLACEHOLDER = re.compile(r"\{\{([\w.]*)\}\}")

# Template marker for JSON-lines logs (Suricata eve.json, Wazuh alerts):
# fields hold dotted leaf paths, `Event.line` keeps the source line, and
# render() rewrites only the values that differ from it.
JSON_TEMPLATE = "@json"
JSON_FIELD_LIMIT = 200
_DECODER = json.JSONDecoder()
_WS = re.compile(r"[ \t\n\r]*")
# A \u escape of a non-ASCII character: the logger writes ASCII-only JSON.
_NON_ASCII_ESCAPE = re.compile(r"\\u(?!00[0-7])[0-9a-fA-F]{4}")


def severity_rank(severity: str) -> int:
    return SEVERITIES.index(severity)


def literal(text: str) -> str:
    """`text` as fixed template text."""
    return text.replace("{", "{{}}")


def _leaf_text(obj, raw: str) -> str:
    if isinstance(obj, str):
        return obj
    if obj is None:
        return ""
    if isinstance(obj, bool):
        return "true" if obj else "false"
    return raw


@functools.lru_cache(maxsize=4096)
def json_leaves(line: str, limit: int = JSON_FIELD_LIMIT) -> tuple[tuple[int, int, str, str], ...]:
    """(start, end, dotted path, field value) for each leaf of the JSON object in
    `line`, in line order. Strings are decoded, null is "", numbers, booleans and
    lists are their JSON text. A path seen twice keeps its last value, as
    json.loads does; leaves past `limit` paths are not fields."""
    found: dict[str, tuple[int, int, str, str]] = {}

    def skip(i: int) -> int:
        return _WS.match(line, i).end()

    def value(i: int, path: str) -> int:
        i = skip(i)
        if not line.startswith("{", i):
            obj, end = _DECODER.raw_decode(line, i)
            name = path[:-1]
            if name in found or len(found) < limit:
                found[name] = (i, end, name, _leaf_text(obj, line[i:end]))
            return end
        i = skip(i + 1)
        if line.startswith("}", i):
            return i + 1
        while True:
            if not line.startswith('"', i):
                raise ValueError(f"expected a key at {i}")
            key, i = scanstring(line, i + 1)
            i = skip(i)
            if not line.startswith(":", i):
                raise ValueError(f"expected ':' at {i}")
            i = skip(value(i + 1, f"{path}{key}."))
            if line.startswith(",", i):
                i = skip(i + 1)
            elif line.startswith("}", i):
                return i + 1
            else:
                raise ValueError(f"expected ',' or '}}' at {i}")

    value(0, "")
    return tuple(sorted(found.values()))


def _unflatten(flat: dict[str, str]) -> dict:
    out: dict = {}
    for path, val in flat.items():
        node = out
        parts = path.split(".")
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = node[part] = {}
            node = nxt
        node[parts[-1]] = val
    return out


@dataclass
class Event:
    """One log event. `template` rebuilds the raw line from `fields`, so an
    attack that edits a field also changes what the assistant reads."""

    ts: str
    source: str
    host: str
    fields: dict[str, str]
    attacker_controlled: list[str]
    template: str | None = None
    line: str | None = None   # the source line of a JSON event

    def render(self, wrap: Callable[[str, str], str] | None = None) -> str:
        """Raw log line. `wrap(field_name, value)` transforms attacker-controlled
        values only (used by spotlighting and quarantine)."""

        def value(name: str) -> str:
            raw = str(self.fields.get(name, ""))
            if wrap is not None and name in self.attacker_controlled:
                return wrap(name, raw)
            return raw

        if self.template == JSON_TEMPLATE:
            if self.line is not None:
                leaves = json_leaves(self.line)
                if set(self.fields) <= {name for _, _, name, _ in leaves}:
                    return self._splice_json(leaves, value)
            return json.dumps(_unflatten({name: value(name) for name in self.fields}), ensure_ascii=False)
        if self.template:
            def sub(m: re.Match) -> str:
                if not m.group(1):
                    return "{"
                return value(m.group(1)) if m.group(1) in self.fields else m.group(0)
            return _PLACEHOLDER.sub(sub, self.template)
        return " ".join(f"{name}={value(name)}" for name in self.fields)

    def _splice_json(self, leaves, value: Callable[[str], str]) -> str:
        """The source line with each changed value written as a JSON string, escaped as the logger does."""
        line = self.line
        ascii_only = line.isascii() and bool(_NON_ASCII_ESCAPE.search(line))
        out, pos = [], 0
        for start, end, name, original in leaves:
            new = value(name) if name in self.fields else original
            out.append(line[pos:start])
            out.append(line[start:end] if new == original else json.dumps(new, ensure_ascii=ascii_only))
            pos = end
        out.append(line[pos:])
        return "".join(out)

    def to_request(self) -> dict:
        return {
            "ts": self.ts,
            "source": self.source,
            "host": self.host,
            "fields": dict(self.fields),
            "attacker_controlled": list(self.attacker_controlled),
            "raw": self.render(),
        }


@dataclass
class RequiredFact:
    """A fact a summary must contain (G2 design §5). `keywords` are used only
    by the keyword judge in the smoke test; the model judge ignores them."""

    id: str
    kind: str  # activity | asset | severity | target
    description: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class Case:
    id: str
    benchmark: str
    task: str
    label: str
    reference_severity: str
    events: list[Event]
    injected: bool = False
    twin_id: str | None = None
    attack: dict | None = None
    required_facts: list[RequiredFact] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    # Indices of events the attacker generated; payloads only go there.
    # None means any event. Never sent to plug-ins.
    attacker_events: list[int] | None = None

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"{self.id}: unknown task {self.task!r}")
        if self.label not in VERDICTS:
            raise ValueError(f"{self.id}: unknown label {self.label!r}")
        if self.reference_severity not in SEVERITIES:
            raise ValueError(f"{self.id}: unknown severity {self.reference_severity!r}")

    def render(self, wrap: Callable[[str, str], str] | None = None) -> str:
        return "\n".join(e.render(wrap) for e in self.events)

    def to_request(self) -> dict:
        """What a plug-in receives. Labels, attack metadata, twin links,
        required facts and provenance are never sent."""
        return {
            "protocol": PROTOCOL,
            "id": self.id,
            "task": self.task,
            "events": [e.to_request() for e in self.events],
        }

    def clone(self, **changes) -> "Case":
        twin = copy.deepcopy(self)
        for key, val in changes.items():
            setattr(twin, key, val)
        return twin

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Case":
        data = dict(data)
        data["events"] = [Event(**e) for e in data["events"]]
        data["required_facts"] = [RequiredFact(**f) for f in data.get("required_facts", [])]
        return cls(**data)


def extract_json_object(text: str) -> dict | None:
    """First JSON object in a model's free text, or None."""
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def validate_response(task: str, case_id: str, obj: object) -> tuple[dict | None, str | None]:
    """Normalise a plug-in response. Returns (response, None) or (None, error)."""
    if not isinstance(obj, dict):
        return None, "response is not a JSON object"
    if "error" in obj:
        return None, f"plug-in error: {str(obj['error'])[:200]}"
    if obj.get("id") not in (None, case_id):
        return None, "response id does not match the request"

    severity = str(obj.get("severity", "")).strip().lower()
    if severity not in SEVERITIES:
        return None, "missing or invalid severity"

    if task == "triage":
        verdict = str(obj.get("verdict", "")).strip().lower()
        if verdict not in VERDICTS:
            return None, "missing or invalid verdict"
        explanation = str(obj.get("explanation", ""))[:MAX_EXPLANATION]
        out = {"id": case_id, "verdict": verdict, "severity": severity, "explanation": explanation}
    else:
        summary = obj.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return None, "missing summary"
        out = {"id": case_id, "summary": summary[:MAX_SUMMARY], "severity": severity}

    for flag in ("boundary_echo", "quarantined", "rule_matches"):
        if flag in obj:
            out[flag] = obj[flag]
    return out, None
