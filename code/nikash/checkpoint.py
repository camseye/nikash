"""The full run's single output file: JSON Lines, appended as work is done.

Each line is one of
    {"kind": "run", "fingerprints": {"cases": ..., "code": ...}, ...}    what the cases and the code were
    {"kind": "record", ...}                          one answer (`nikash.run._record`)
    {"kind": "section", "name": ..., "data": ...}    a report; the last one per name counts

Answers are appended in slices and flushed to disk, so a session that stops
keeps everything up to its last slice. A line cut off by a stopped session is
skipped when the file is read. A new session resumes from earlier output files
with the same fingerprints: their answers are copied into this file and not
asked again. Files with other cases, or answers from other code or generation
settings, are refused (`Checkpoint.refused` says which) and never mixed in.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .schema import Case

PACKAGE = Path(__file__).resolve().parent
WHAT = {"cases": "cases", "code": "code or generation settings"}


def case_fingerprint(cases: list[Case]) -> str:
    """Changes whenever any case's id or rendered window changes."""
    h = hashlib.sha256()
    for c in sorted(cases, key=lambda c: c.id):
        h.update(c.id.encode("utf-8") + b"\0" + c.render().encode("utf-8") + b"\0")
    return h.hexdigest()


def code_fingerprint(settings: dict, root: Path = PACKAGE) -> str:
    """Changes whenever any .py file under `root` (the nikash package) or any of `settings` changes.
    Line endings are normalised, so a Windows and a Linux checkout of the same commit agree."""
    root = Path(root)
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda p: p.relative_to(root).as_posix()):
        text = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        h.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0" + hashlib.sha256(text).digest())
    h.update(json.dumps(settings, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def run_fingerprints(cases: list[Case], settings: dict, root: Path = PACKAGE) -> dict[str, str]:
    """`settings`: everything outside the code that changes an answer (model ids, generation settings)."""
    return {"cases": case_fingerprint(cases), "code": code_fingerprint(settings, root)}


def mismatch(run: dict | None, fingerprints: dict[str, str]) -> str | None:
    """Why the run line `run` is not this run, or None when it is."""
    if run is None:
        return "no run line"
    theirs = run.get("fingerprints") or {}
    differ = [k for k in fingerprints if theirs.get(k) != fingerprints[k]]
    return f"different {' and '.join(WHAT.get(k, k) for k in differ)}" if differ else None


def case_stub(case: Case) -> dict:
    """What the scorer needs about a case, without its log text or payload."""
    return {"id": case.id, "benchmark": case.benchmark, "task": case.task, "label": case.label,
            "injected": case.injected, "twin_id": case.twin_id, "reference_severity": case.reference_severity,
            "cluster": case.provenance.get("cluster"), "incident": case.provenance.get("incident"),
            "level": (case.attack or {}).get("level")}


def read_lines(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue   # cut off when a session stopped mid-write
            if isinstance(obj, dict) and "kind" in obj:
                out.append(obj)
    return out


class Checkpoint:
    def __init__(self, path: Path, fingerprints: dict[str, str], run_info: dict) -> None:
        """`fingerprints`: from `run_fingerprints`; a file is resumed only if all of them match."""
        self.path = Path(path)
        self.fingerprints = dict(fingerprints)
        self.done: dict[tuple[str, str, str], dict] = {}
        self.sections: dict[str, object] = {}
        self.resumed: dict[str, int] = {}
        self.refused: dict[str, str] = {}    # path -> why
        self.earlier: dict[str, dict] = {}   # sections of the files resumed from, by path
        self.earlier_runs: dict[str, dict] = {}   # their run lines, by path
        if self.path.exists():
            lines = read_lines(self.path)
            why = mismatch(next((l for l in lines if l["kind"] == "run"), None), self.fingerprints)
            if why:
                raise ValueError(f"{self.path} holds another run ({why}); move it away first")
            self._take(lines)
            self._end_line()
        else:
            self._write([{"kind": "run", "fingerprints": self.fingerprints, **run_info}])

    @staticmethod
    def key(record: dict) -> tuple[str, str, str]:
        return record["model"], record["defence"], record["case_id"]

    def _take(self, lines: list[dict]) -> list[dict]:
        new = []
        for line in lines:
            if line["kind"] == "record":
                record = {k: v for k, v in line.items() if k != "kind"}
                if self.key(record) not in self.done:
                    self.done[self.key(record)] = record
                    new.append(line)
            elif line["kind"] == "section":
                self.sections[line["name"]] = line["data"]
        return new

    def _end_line(self) -> None:
        # A stopped session may have left a partial last line; the next line must start fresh.
        with open(self.path, "rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            if size:
                fh.seek(size - 1)
            last = fh.read(1) if size else b"\n"
        if last != b"\n":
            with open(self.path, "a", encoding="utf-8") as out:
                out.write("\n")

    def _write(self, lines: list[dict]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            for line in lines:
                fh.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def resume_from(self, paths: list[Path]) -> None:
        """Copies in the answers of earlier output files made from the same cases, code and settings."""
        for path in paths:
            if Path(path).resolve() == self.path.resolve():
                continue
            lines = read_lines(Path(path))
            run = next((l for l in lines if l["kind"] == "run"), None)
            why = mismatch(run, self.fingerprints)
            if why:
                self.refused[str(path)] = why
                continue
            new = self._take([l for l in lines if l["kind"] == "record"])
            self._write(new)
            self.resumed[str(path)] = len(new)
            self.earlier[str(path)] = {l["name"]: l["data"] for l in lines if l["kind"] == "section"}
            self.earlier_runs[str(path)] = run

    def changed(self, name: str, value) -> dict[str, object]:
        """The resumed files whose run line has another `name` than `value` (a missing one counts), with
        theirs. For settings outside the fingerprints, such as library versions: worth a warning, not a refusal."""
        return {path: run.get(name) for path, run in self.earlier_runs.items() if run.get(name) != value}

    def add(self, records: list[dict]) -> None:
        lines = [{"kind": "record", **r} for r in records if self.key(r) not in self.done]
        for line in lines:
            record = {k: v for k, v in line.items() if k != "kind"}
            self.done[self.key(record)] = record
        self._write(lines)

    def section(self, name: str, data) -> None:
        self.sections[name] = data
        self._write([{"kind": "section", "name": name, "data": data}])
        print(f"Saved section '{name}' in {self.path} ({self.path.stat().st_size:,} bytes)")

    def todo(self, model: str, defence: str, cases: list[Case]) -> list[Case]:
        return [c for c in cases if (model, defence, c.id) not in self.done]

    def records(self) -> list[dict]:
        return list(self.done.values())
