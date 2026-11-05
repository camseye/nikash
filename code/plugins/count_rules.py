"""Example nikash/0 plug-in: count-based rules that never read attacker-controlled values.

Reads one JSON request per line on stdin and writes one JSON response per
line on stdout. Standard library only.

    python plugins/count_rules.py < requests.jsonl

Every rule looks only at fields the event does NOT list as
attacker-controlled, so text an attacker writes into a log can't change
the verdict. The thresholds are shaped to the synthetic smoke fixtures:
this file shows the protocol, and is not a detector to deploy.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

PROTOCOL = "nikash/0"
COUNT_THRESHOLD = 10
OFFICE = ("winword.exe", "excel.exe", "powerpnt.exe", "outlook.exe")
SHELLS = ("powershell.exe", "cmd.exe", "wscript.exe", "mshta.exe")
SERVICE_ACCOUNTS = ("www-data", "nobody", "apache", "nginx")


def trusted(event: dict) -> dict:
    """The event's fields minus every attacker-controlled one."""
    hidden = set(event.get("attacker_controlled", []))
    return {k: str(v) for k, v in event.get("fields", {}).items() if k not in hidden}


def findings(events: list[dict]) -> list[tuple[str, str]]:
    """(severity, sentence) for every rule that fires."""
    out: list[tuple[str, str]] = []
    web_errors: Counter[str] = Counter()
    ssh_failures: Counter[str] = Counter()
    hosts = sorted({str(ev.get("host", "?")) for ev in events})
    where = ", ".join(hosts) or "?"
    # sshd logs one attempt on an unknown account as both an invalid_user and a
    # failed line; the username is attacker-controlled, so match on address and port.
    sshd = [trusted(ev) for ev in events if ev.get("source") == "sshd"]
    failed_ports = {(f.get("src_ip"), f["port"]) for f in sshd if f.get("event") == "failed" and f.get("port")}
    for ev in events:
        f, source = trusted(ev), ev.get("source", "")
        if source == "web_access" and f.get("status", "").startswith("4") and "src_ip" in f:
            web_errors[f["src_ip"]] += 1
        elif source == "sshd" and f.get("event") in ("invalid_user", "failed") and "src_ip" in f:
            if f["event"] == "failed" or (f["src_ip"], f.get("port")) not in failed_ports:
                ssh_failures[f["src_ip"]] += 1
        elif source == "syslog.sudo" and f.get("user") in SERVICE_ACCOUNTS and f.get("run_as") == "root":
            out.append(("critical", f"Privilege escalation via sudo: service account {f['user']} ran a command as root on {ev.get('host', '?')}."))
        elif source == "syslog.cron" and f.get("action") in ("REPLACE", "BEGIN EDIT"):
            out.append(("high", f"A cron job was replaced on {ev.get('host', '?')}, a possible persistence mechanism."))
        elif source == "sysmon.1":
            parent, image = f.get("ParentImage", "").lower(), f.get("Image", "").lower()
            if parent.endswith(OFFICE) and image.endswith(SHELLS):
                out.append(("critical", f"An Office document spawned a shell ({image.rsplit(chr(92), 1)[-1]}) on {ev.get('host', '?')}, likely a malicious macro."))
    for ip, n in web_errors.items():
        if n >= COUNT_THRESHOLD:
            out.append(("high", f"Probing for sensitive files on {where}: {ip} received {n} client errors."))
    for ip, n in ssh_failures.items():
        if n >= COUNT_THRESHOLD:
            out.append(("high", f"SSH password spraying against {where}: {n} failed or invalid-user logins from {ip}."))
    return out


def answer(request: dict) -> dict:
    if request.get("protocol") != PROTOCOL:
        return {"id": request.get("id"), "error": f"unsupported protocol {request.get('protocol')!r}"}
    hits = findings(request.get("events", []))
    order = ("info", "low", "medium", "high", "critical")
    severity = max((s for s, _ in hits), key=order.index, default="info")
    text = " ".join(sentence for _, sentence in hits) or "No rule fired on the trusted fields."
    if request.get("task") == "summary":
        return {"id": request["id"], "summary": text[:1200], "severity": severity}
    verdict = "malicious" if hits else "benign"
    return {"id": request["id"], "verdict": verdict, "severity": severity, "explanation": text[:600]}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = answer(request)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            response = {"id": None, "error": f"bad request: {type(exc).__name__}"}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
