"""Run an external plug-in over the nikash/0 protocol (JSON Lines on stdin/stdout).

The command is a list of arguments and is started without a shell. The
plug-in receives only `Case.to_request()`, never labels or attack data.

A case that times out fails, and the plug-in is killed and restarted for
the next one, so a late reply can't be taken as another case's answer. A
reply whose `id` names another case is invalid. Once the plug-in exits or
closes its input, every remaining case fails with that reason.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading

from .schema import Case, validate_response


class ExternalPlugin:
    model_decides = True

    def __init__(self, command: list[str], *, name: str, timeout: float = 60.0) -> None:
        if not isinstance(command, list) or not command:
            raise ValueError("command must be a non-empty list of arguments")
        self.command = command
        self.id = name
        self.name = name
        self.timeout = timeout
        self.restarts = 0
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] | None = None
        self._dead: str | None = None

    def __enter__(self) -> "ExternalPlugin":
        self._dead = None
        self._start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop()
        self._dead = None

    def _start(self) -> None:
        """A fresh process with its own reply queue, so no line from an earlier
        process (a late reply after a timeout, or an earlier run) is ever read."""
        proc = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
        )
        lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, args=(proc, lines), daemon=True).start()
        self._proc, self._lines = proc, lines

    def _stop(self, kill: bool = False) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if kill:
                proc.kill()
            elif proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
            proc.wait()
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except OSError:
                pass

    @staticmethod
    def _pump(proc: subprocess.Popen, lines: queue.Queue) -> None:
        assert proc.stdout is not None
        try:
            with proc.stdout:
                for line in proc.stdout:
                    lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            lines.put(None)

    def ask(self, case: Case) -> tuple[dict | None, str | None, str]:
        if self._dead is not None:
            return None, self._dead, ""
        if self._proc is None or self._proc.stdin is None or self._lines is None:
            raise RuntimeError("plug-in is not running; use it as a context manager")
        try:
            self._proc.stdin.write(json.dumps(case.to_request()) + "\n")
            self._proc.stdin.flush()
        except OSError:
            self._dead = "plug-in closed its input earlier in the run"
            return None, "plug-in closed its input", ""
        try:
            line = self._lines.get(timeout=self.timeout)
        except queue.Empty:
            # A late reply would otherwise become the next case's answer.
            self._stop(kill=True)
            try:
                self._start()
                self.restarts += 1
            except OSError:
                self._dead = "plug-in could not be restarted after a timeout"
            return None, "timeout (plug-in restarted)" if self._dead is None else "timeout", ""
        if line is None:
            self._dead = "plug-in exited earlier in the run"
            return None, "plug-in exited", ""
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None, "response is not valid JSON", line[:2000]
        if isinstance(obj, dict) and obj.get("id") is not None and obj["id"] != case.id:
            return None, "response id does not match the request", line[:2000]
        response, error = validate_response(case.task, case.id, obj)
        return response, error, line[:2000]


def run_plugin(cases: list[Case], plugin: ExternalPlugin) -> list[dict]:
    records = []
    with plugin:
        for case in cases:
            response, error, text = plugin.ask(case)
            records.append({
                "case_id": case.id, "benchmark": case.benchmark, "task": case.task,
                "defence": plugin.id, "model": "external", "ok": response is not None,
                "error": error, "response": response, "model_text": text,
            })
    return records
