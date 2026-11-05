"""Raw log line → Event, for the formats found in the public benchmarks.

Each parser keeps the original line as a template and swaps only the
captured spans for `{{field}}` placeholders, so `Event.render()` with no
edits returns the line unchanged and an attack edit changes nothing else.
JSON events keep the line itself and rewrite only the values that change.

Lines no parser recognises become `generic` events with no
attacker-controlled fields: they are shown to the model but never injected.
"""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath

from ..schema import JSON_TEMPLATE, Event, json_leaves, literal

_APACHE = re.compile(
    r'^(?P<src_ip>\S+) \S+ (?P<auth_user>\S+) \[(?P<ts>[^\]]+)\] '
    r'"(?P<method>[A-Z]+) (?P<uri>\S+)(?: [^"]*)?" (?P<status>\d{3}) (?P<bytes>\S+)'
    r'(?: "(?P<referer>(?:[^"\\]|\\.)*)" "(?P<user_agent>(?:[^"\\]|\\.)*)")?'
)
# A host never holds "[" or ends in ":", so "sshd[1]:" after a stamp is a program.
_SYSLOG = re.compile(
    r"^(?P<ts>[A-Z][a-z]{2} [ \d]\d \d\d:\d\d:\d\d|\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?) "
    r"(?P<host>[^\s\[]*[^\s\[:]) (?P<prog>[\w./()-]+?)(?:\[(?P<pid>\d+)\])?: "
)
# OpenSSH 9.8 moved per-connection logging to `sshd-session`.
_SSHD_PROGS = {"sshd", "sshd-session"}
# No host field, as LogInject-1.0 writes SSH lines: "2025-01-11T18:41:15 sshd[2594]: ...".
_SYSLOG_NO_HOST = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?) (?P<host>)(?P<prog>[\w./-]+)\[(?P<pid>\d+)\]: "
)
# A client chooses the username, which may itself contain " from ": the
# source address is the last "from" in the line.
_SSHD = [
    ("invalid_user", re.compile(r"Invalid user (?P<username>.*) from (?P<src_ip>\S+)(?: port (?P<port>\d+))?")),
    ("failed", re.compile(r"Failed (?:password|publickey|none) for (?:invalid user )?(?P<username>.*) from (?P<src_ip>\S+)(?: port (?P<port>\d+))?")),
    ("accepted", re.compile(r"Accepted \S+ for (?P<username>\S+) from (?P<src_ip>\S+)(?: port (?P<port>\d+))?")),
]
_SUDO = re.compile(r"\s*(?P<user>\S+) : .*?USER=(?P<run_as>\S+) ; COMMAND=(?P<command>.*)$")
_CRON_EXEC = re.compile(r"\((?P<user>[^)]+)\) CMD \((?P<command>.*)\)$")
_CRONTAB = re.compile(r"\((?P<user>[^)]+)\) (?P<action>[A-Z]+(?: [A-Z]+)*) \((?P<cron_name>[^)]*)\)")

# Fields in JSON-lines logs that a remote party chooses.
_JSON_CONTROLLED = {
    "http.http_user_agent", "http.http_refer", "http.url", "http.hostname",
    "dns.rrname", "dns.query.rrname", "tls.sni", "full_log",
}


def _splice(line: str, match: re.Match, names: list[str]) -> tuple[dict[str, str], str]:
    """Fields from the named groups, and the line with those spans templated."""
    fields: dict[str, str] = {}
    spans = []
    for name in names:
        if match.group(name) is None:
            continue
        fields[name] = match.group(name)
        spans.append((match.start(name), match.end(name), name))
    out, pos = [], 0
    for start, end, name in sorted(spans):
        out.append(literal(line[pos:start]))
        out.append("{{" + name + "}}")
        pos = end
    out.append(literal(line[pos:]))
    return fields, "".join(out)


def _json_event(line: str, host: str, path: str) -> Event | None:
    try:
        obj = json.loads(line)
        leaves = json_leaves(line)
    except (ValueError, RecursionError):
        return None
    if not isinstance(obj, dict):
        return None
    fields = {name: val for _, _, name, val in leaves}
    if "event_type" in obj:
        source = f"suricata.{obj['event_type']}"
        ts = str(obj.get("timestamp", ""))
    elif "rule" in obj and "agent" in obj:
        source = "wazuh"
        ts = str(obj.get("timestamp", ""))
    else:
        source = f"json.{PurePosixPath(path).stem}"
        ts = str(obj.get("timestamp", obj.get("@timestamp", "")))
    return Event(ts=ts, source=source, host=host, fields=fields,
                 attacker_controlled=[k for k in fields if k in _JSON_CONTROLLED], template=JSON_TEMPLATE, line=line)


def _syslog_event(line: str, m: re.Match, host: str) -> Event:
    prog = m.group("prog").lower()
    ts = m.group("ts")
    rest = m.end()
    log_host = m.group("host") or host

    if prog in _SSHD_PROGS:
        for kind, pattern in _SSHD:
            sm = pattern.search(line, rest)
            if sm:
                fields, template = _splice(line, sm, ["username", "src_ip", "port"])
                fields["event"] = kind
                return Event(ts, "sshd", log_host, fields,
                             [] if kind == "accepted" else ["username"], template)
    elif prog == "sudo":
        sm = _SUDO.match(line, rest)
        if sm:
            fields, template = _splice(line, sm, ["user", "run_as", "command"])
            return Event(ts, "syslog.sudo", log_host, fields, [], template)
    elif prog == "cron":
        sm = _CRON_EXEC.match(line, rest)
        if sm:
            fields, template = _splice(line, sm, ["user", "command"])
            return Event(ts, "syslog.cron_exec", log_host, fields, ["command"], template)
    elif prog == "crontab":
        sm = _CRONTAB.match(line, rest)
        if sm:
            fields, template = _splice(line, sm, ["user", "action", "cron_name"])
            return Event(ts, "syslog.cron", log_host, fields, ["cron_name"], template)
    # Separate namespace: names like `syslog.cron` carry meaning in checks.py.
    return Event(ts, f"syslog.other.{prog}", log_host, {"message": line[rest:]}, [], literal(line[:rest]) + "{{message}}")


def parse_line(line: str, host: str, path: str) -> Event:
    """`path` is the log file's path relative to the host directory; it is
    used only to name the source of unrecognised lines."""
    line = line.rstrip("\r\n")
    if line.lstrip().startswith("{"):
        ev = _json_event(line, host, path)
        if ev is not None:
            return ev
    m = _APACHE.match(line)
    if m:
        fields, template = _splice(line, m, ["src_ip", "auth_user", "method", "uri", "status", "bytes", "referer", "user_agent"])
        controlled = [k for k in ("uri", "referer", "user_agent") if k in fields]
        return Event(m.group("ts"), "web_access", host, fields, controlled, template)
    m = _SYSLOG.match(line) or _SYSLOG_NO_HOST.match(line)
    if m:
        return _syslog_event(line, m, host)
    name = PurePosixPath(path).name.split(".")[0] or "log"
    return Event("", f"generic.{name}", host, {"message": line}, [], "{{message}}")
