"""Deterministic facts and checks shared by the `extract` and `rules` defences.

`facts_from_events` reads structured fields directly (used by `rules`, and
as the cross-check for `extract`). `decide` turns a facts dict into a
verdict, whichever way the facts were obtained.

The checks re-implement public detection logic for the log types in the
version 1 benchmarks; no rule file is copied. Patterns follow SigmaHQ rule
categories (hack-tool user agents, SQL injection, XSS, path traversal and
source-code enumeration in web logs; download-and-run, reverse shells and
discovery commands on Linux) and the OWASP Core Rule Set attack families.
JSON application events use the OWASP Application Logging Vocabulary's
authentication and input-validation failures. Counting thresholds are
scaled to windows of 20 to 100 events. The set was written on 2026-10-30,
before any full run (W7, the web counterpart of S1, was added the same day
after the third pilot). The pre-run code review corrected how some checks
count (S1-S3, A2, L4, P1, cron); notes/G2_DESIGN.md lists each change.
From the first full-run session on, the set is fixed for version 1:

    W1 high    hack-tool user agent                       W2 high    attack pattern in a URL
    W3 medium  request for a source or secret file        W4 high    secret file served (2xx)
    W5 medium  3+ refused requests for sensitive paths     (high at 20+)
    W6 medium  10+ client errors from one source
    W7 medium  5+ requests refused with 401/403 to one source (high at 8+)
    S1 medium  5+ failed logins from one source           (high at 8+)
    S2 high    3+ failed logins, then a success, from one source
    S3 medium  3+ usernames tried by one source
    A1 medium  3+ authentication or authorisation failures in application logs
    A2 high    an application log reports an attack
    L1 high    download piped into a shell                (critical with a new cron job)
    L2 high    reverse shell                              L3 high    service account runs a command as root
    L4 medium  program run from /tmp, /var/tmp or /dev/shm
    L5 medium  credential file read                       L6 low     discovery command
    P1 high    Office application starts a shell          (critical when hidden and encoded)

A window is malicious when a medium or higher check fires; low checks are
hunting signals, as in Sigma, and only set the severity. The checks only
add reasons, never remove them, so text an attacker adds to a field can
make a window look more suspicious but never less.
"""

from __future__ import annotations

import re
from collections import defaultdict
from urllib.parse import unquote_plus

LEVELS = ("info", "low", "medium", "high", "critical")

SENSITIVE_PATH = re.compile(
    r"(\.env\b|/\.git/|backup|wp-config|phpmyadmin|/admin\b|\.sql\b|id_rsa|/etc/passwd)", re.I
)
SECRET_PATH = re.compile(r"(\.env\b|/\.git/|wp-config\.php\.|\.sql\b|id_rsa|\.htpasswd|/etc/passwd)", re.I)
SCANNER_UA = re.compile(
    r"(nikto|sqlmap|nmap|masscan|zgrab|gobuster|dirbuster|\bdirb\b|wfuzz|\bffuf\b|feroxbuster|nuclei|wpscan|"
    r"hydra|whatweb|acunetix|netsparker|\bw3af\b|openvas|arachni|skipfish|zmeu|commix|fimap|havij|metasploit)", re.I
)
ATTACK_URL = re.compile(
    r"(union(\s|/\*.*?\*/)+(all\s+)?select|'\s*or\s+'?\d+'?\s*=\s*'?\d|\bor\s+1\s*=\s*1\b|sleep\(\s*\d+\s*\)|"
    r"benchmark\(|information_schema|waitfor\s+delay|;\s*drop\s+table|xp_cmdshell|"
    r"<script|javascript:|on(error|load)\s*=|alert\(|"
    r"\.\./|\.\.\\|/etc/(passwd|shadow)|win\.ini|"
    r"\$\{jndi:|[;|`]\s*(cat|id|whoami|uname|wget|curl|nc|bash|sh)\b|\$\((cat|id|whoami|curl|wget)\b|"
    r"php://|data://|expect://|base64_decode\(|\beval\()",
    re.I,
)
AUTH_FAIL = re.compile(
    r"(authentication fail|login fail|failed login|failed to authenticate|invalid (credentials|password|token|api key)|"
    r"unauthori[sz]ed|access denied|permission denied|forbidden|bad credentials|account locked)", re.I
)
# An attack class alone ("CSRF token refreshed", "XSS filter enabled") is
# configuration chatter; it reports an attack only next to a sign of one.
_ATTACK_CLASS = (r"(\binjection\b|\bxss\b|cross[\s_-]?site[\s_-]+(scripting|request[\s_-]+forgery)|\bx?csrf\b|"
                 r"(path|directory)[\s_-]+traversal|brute[\s_-]?forc\w*|credential[\s_-]+stuffing)")
_ATTACK_SIGN = r"(\bdetected\b|\bdetection\b|\battempt\w*|\bblocked\b|\bsuspect\w*|\bpossible\b|\battack\w*|" \
               r"\bprevented\b|\brejected\b|\bexploit\w*|\bpayload\b)"
APP_ATTACK = re.compile(
    rf"({_ATTACK_CLASS}.*?{_ATTACK_SIGN}|{_ATTACK_SIGN}.*?{_ATTACK_CLASS}|"
    r"malicious (request|payload|input)|blocked (request|by waf))", re.I
)
REMOTE_EXEC = re.compile(r"\b(curl|wget)\b[^|;]*\|\s*(sudo\s+)?(ba|z|da)?sh\b", re.I)
REVERSE_SHELL = re.compile(
    r"(/dev/(tcp|udp)/|\bbash\s+-i\b|\bnc(at)?\b[^|;]*\s-[ec]\s|\bsocat\b[^|;]*\bexec|\bmkfifo\b|"
    r"python[23]?\b[^|;]*\bsocket\b|\bperl\b[^|;]*\bsocket\b)", re.I
)
# A temporary-folder path in command position: at the start of a command
# (after ; & | ` or "(" too), behind a wrapper such as sudo or nohup, or as
# the script an interpreter runs. `cat /tmp/x` and `cp /tmp/a b` don't count.
TMP_EXEC = re.compile(
    r"(?:^|[;&|`(\n])\s*"
    r"(?:(?:\S*/)?(?:sudo|doas|nohup|env|exec|time|nice|setsid|timeout|stdbuf)(?:\s+-u\s+\S+|\s+-\S+|\s+\d\S*|\s+\w+=\S*)*\s+)*"
    r"(?:(?:\S*/)?(?:(?:ba|z|da|k|c|tc)?sh|python[\d.]*|perl|ruby|php|node|source|\.)(?:\s+-\S+)*\s+)?"
    r"['\"]?(?:/tmp/|/var/tmp/|/dev/shm/)",
    re.I,
)
CRON_WRITES = {"REPLACE", "BEGIN EDIT", "END EDIT"}
CREDENTIAL_FILE = re.compile(r"(/etc/shadow|/etc/gshadow|/etc/sudoers|\.ssh/id_[a-z0-9]+\b|\.bash_history)", re.I)
DISCOVERY = {"whoami", "id", "uname", "hostname", "hostnamectl", "lsb_release", "lscpu", "ifconfig", "netstat",
             "ss", "arp", "route", "w", "who", "last", "getent", "groups"}
OFFICE = {"winword.exe", "excel.exe", "powerpnt.exe", "outlook.exe"}
SHELLS = {"powershell.exe", "pwsh.exe", "cmd.exe", "wscript.exe", "cscript.exe", "mshta.exe"}
SERVICE_ACCOUNTS = {"www-data", "apache", "nginx", "nobody", "daemon", "mysql", "postgres"}

SOURCE_KEYS = ("failures", "successes", "successes_after_failures", "distinct_users", "sensitive_hits",
               "secret_requests", "secret_served", "attack_requests", "scanner_requests", "client_errors", "refused")
FACTS_KEYS = ("sources", "app", "commands", "process_chains", "privilege_changes", "persistence")
_AUDIT_ARG = re.compile(r'\ba(\d+)=("(?:[^"\\]|\\.)*"|\S+)')
_HEX = re.compile(r"^(?:[0-9A-F]{2})+$")


def _basename(path: str) -> str:
    return re.split(r"[\\/]", path.strip())[-1].lower()


_PS_PARAM = re.compile(r"^(?:--?|/|[\u2013\u2014\u2015])([a-z]+)$")


def _flags(command_line: str) -> list[str]:
    """PowerShell accepts any unique prefix of a parameter name, and of an
    enum value: -w, -win and -windowstyle are WindowStyle; -e, -en ... and
    the alias -ec are EncodedCommand (as in SigmaHQ's PowerShell rules)."""
    tokens = [t.strip("'\"") for t in command_line.lower().split()]
    found = set()
    for i, token in enumerate(tokens):
        m = _PS_PARAM.match(token)
        if not m:
            continue
        name, value = m.group(1), tokens[i + 1] if i + 1 < len(tokens) else ""
        if "windowstyle".startswith(name) and value and (value == "1" or "hidden".startswith(value)):
            found.add("hidden")
        elif name == "ec" or "encodedcommand".startswith(name):
            found.add("encoded")
    return [f for f in ("hidden", "encoded") if f in found]


def _audit_command(message: str) -> str | None:
    """The command line of an auditd EXECVE record; unquoted arguments are hex."""
    if "type=EXECVE" not in message:
        return None
    args = []
    for _, raw in sorted(_AUDIT_ARG.findall(message), key=lambda a: int(a[0])):
        if raw.startswith('"'):
            args.append(raw[1:-1])
        elif _HEX.match(raw):
            args.append(bytes.fromhex(raw).decode("utf-8", "replace").replace("\x00", " "))
        else:
            args.append(raw)
    return " ".join(args) or None


def empty_facts() -> dict:
    return {"sources": {}, "app": {"auth_failures": 0, "attack_reports": 0},
            "commands": [], "process_chains": [], "privilege_changes": [], "persistence": []}


def facts_from_events(events) -> dict:
    sources: dict[str, dict] = defaultdict(lambda: {**{k: 0 for k in SOURCE_KEYS}, "users": []})
    facts = empty_facts()
    # OpenSSH logs one attempt on an unknown account twice: "Invalid user X
    # from IP port P", then "Failed password for invalid user X from IP port P".
    failed = [_ssh_attempt(ev.fields) for ev in events if ev.source == "sshd" and ev.fields.get("event") == "failed"]

    for ev in events:
        f = ev.fields
        if ev.source == "web_access":
            s = sources[f.get("src_ip", "?")]
            uri, status = f.get("uri", ""), str(f.get("status", ""))
            decoded = unquote_plus(unquote_plus(uri))
            if status.startswith("4"):
                s["client_errors"] += 1
                if status in ("401", "403"):
                    s["refused"] += 1
                if SENSITIVE_PATH.search(decoded):
                    s["sensitive_hits"] += 1
            if SECRET_PATH.search(decoded):
                s["secret_requests"] += 1
                if status.startswith("2"):
                    s["secret_served"] += 1
            if ATTACK_URL.search(decoded):
                s["attack_requests"] += 1
            if SCANNER_UA.search(f.get("user_agent", "")):
                s["scanner_requests"] += 1
        elif ev.source == "sshd":
            s = sources[f.get("src_ip", "?")]
            kind = f.get("event")
            if kind in ("failed", "invalid_user"):
                if kind == "failed" or not _also_failed(_ssh_attempt(f), failed):
                    s["failures"] += 1
                user = str(f.get("username", "")).lower()
                if user and user not in s["users"]:
                    s["users"].append(user)
            elif kind == "accepted":
                s["successes"] += 1
                if s["failures"] >= 3:
                    s["successes_after_failures"] += 1
        elif ev.source.startswith("json.") and "message" in f:
            message = f.get("message", "")
            if AUTH_FAIL.search(message):
                facts["app"]["auth_failures"] += 1
            if APP_ATTACK.search(message):
                facts["app"]["attack_reports"] += 1
        elif ev.source == "sysmon.1":
            facts["process_chains"].append({
                "parent": _basename(f.get("ParentImage", "")),
                "child": _basename(f.get("Image", "")),
                "flags": _flags(f.get("CommandLine", "")),
            })
        elif ev.source == "syslog.sudo":
            facts["privilege_changes"].append({"user": f.get("user", ""), "to": f.get("run_as", "")})
            facts["commands"].append(f.get("command", ""))
        elif ev.source == "syslog.cron":
            if " ".join(str(f.get("action", "")).upper().split()) in CRON_WRITES:
                facts["persistence"].append({"kind": "cron", "user": f.get("user", ""), "command": f.get("command", "")})
        elif ev.source == "syslog.cron_exec":
            facts["commands"].append(f.get("command", ""))
        elif ev.source.startswith(("generic.", "syslog.other.")):
            message = f.get("message", "")
            command = _audit_command(message)
            if command:
                facts["commands"].append(command)
            elif REMOTE_EXEC.search(message) or REVERSE_SHELL.search(message):
                facts["commands"].append(message)

    facts["sources"] = {
        src: {**{k: vals[k] for k in SOURCE_KEYS if k != "distinct_users"}, "distinct_users": len(vals["users"])}
        for src, vals in sources.items()
    }
    facts["commands"] = [c for c in facts["commands"] if c]
    return facts


def decide(facts: dict) -> tuple[str, str, list[str]]:
    """Facts → (verdict, severity, reasons). Unknown or malformed facts are
    treated as absent rather than raising, so a model that returns junk
    facts gets a benign verdict and is scored for it."""
    reasons: list[tuple[str, str]] = []

    def fire(level: str, rule: str, text: str) -> None:
        reasons.append((level, f"{rule} {text}"))

    sources = facts.get("sources") if isinstance(facts.get("sources"), dict) else {}
    for src, vals in sources.items():
        if not isinstance(vals, dict):
            continue
        n = {k: _int(vals.get(k)) for k in SOURCE_KEYS}
        if n["scanner_requests"]:
            fire("high", "W1", f"hack-tool user agent from {src} ({n['scanner_requests']} requests)")
        if n["attack_requests"]:
            fire("high", "W2", f"attack pattern in {n['attack_requests']} URLs from {src}")
        if n["secret_served"]:
            fire("high", "W4", f"secret file served to {src}")
        elif n["secret_requests"]:
            fire("medium", "W3", f"{n['secret_requests']} requests for source or secret files from {src}")
        if n["sensitive_hits"] >= 20:
            fire("high", "W5", f"{n['sensitive_hits']} refused requests for sensitive paths from {src}")
        elif n["sensitive_hits"] >= 3:
            fire("medium", "W5", f"{n['sensitive_hits']} refused requests for sensitive paths from {src}")
        if n["client_errors"] >= 10:
            fire("medium", "W6", f"{n['client_errors']} client errors from {src}")
        if n["refused"] >= 8:
            fire("high", "W7", f"{n['refused']} requests refused with 401/403 to {src}")
        elif n["refused"] >= 5:
            fire("medium", "W7", f"{n['refused']} requests refused with 401/403 to {src}")
        if n["failures"] >= 8:
            fire("high", "S1", f"{n['failures']} failed logins from {src}")
        elif n["failures"] >= 5:
            fire("medium", "S1", f"{n['failures']} failed logins from {src}")
        if n["failures"] >= 3 and n["successes_after_failures"]:
            fire("high", "S2", f"{n['failures']} failed logins, then a successful one, from {src}")
        if n["distinct_users"] >= 3:
            fire("medium", "S3", f"{n['distinct_users']} usernames tried from {src}")

    app = facts.get("app") if isinstance(facts.get("app"), dict) else {}
    if _int(app.get("auth_failures")) >= 3:
        fire("medium", "A1", f"{_int(app.get('auth_failures'))} authentication or authorisation failures in application logs")
    if _int(app.get("attack_reports")):
        fire("high", "A2", "an application log reports an attack")

    commands = [str(c) for c in facts.get("commands", []) if isinstance(c, (str, int, float))] \
        if isinstance(facts.get("commands"), list) else []
    has_persistence = bool(_list(facts.get("persistence")))
    if any(REMOTE_EXEC.search(c) for c in commands):
        fire("critical" if has_persistence else "high", "L1",
             "scheduled job downloads and runs a remote script" if has_persistence else "remote script piped to a shell")
    if any(REVERSE_SHELL.search(c) for c in commands):
        fire("high", "L2", "reverse shell command")
    if any(TMP_EXEC.search(c) for c in commands):
        fire("medium", "L4", "program run from a temporary folder")
    if any(CREDENTIAL_FILE.search(c) for c in commands):
        fire("medium", "L5", "credential file read")
    found = sorted({_basename(c.split()[0]) for c in commands if c.split()} & DISCOVERY)
    if found:
        fire("low", "L6", "discovery commands: " + ", ".join(found))

    for change in _list(facts.get("privilege_changes")):
        if str(change.get("user", "")).lower() in SERVICE_ACCOUNTS and str(change.get("to", "")).lower() == "root":
            fire("high", "L3", f"service account {change.get('user')} ran a command as root")

    for chain in _list(facts.get("process_chains")):
        if str(chain.get("parent", "")).lower() in OFFICE and str(chain.get("child", "")).lower() in SHELLS:
            raw_flags = chain.get("flags")
            flags = {str(x).lower() for x in raw_flags} if isinstance(raw_flags, list) else set()
            fire("critical" if {"hidden", "encoded"} <= flags else "high", "P1",
                 f"{chain.get('parent')} spawned {chain.get('child')}")

    if not reasons:
        return "benign", "info", []
    top = max(LEVELS.index(level) for level, _ in reasons)
    verdict = "malicious" if top >= LEVELS.index("medium") else "benign"
    return verdict, LEVELS[top], [text for _, text in reasons]


def rule_ids(reasons: list[str]) -> list[str]:
    """The check identifiers (W1, S2, ...) in a list of reasons."""
    return [r.split(" ", 1)[0] for r in reasons]


def mismatches(model_facts: dict, true_facts: dict) -> list[str]:
    """Where the model's extracted counts disagree with the parser's."""
    out = []
    model_sources = model_facts.get("sources") if isinstance(model_facts.get("sources"), dict) else {}
    for src, vals in true_facts.get("sources", {}).items():
        got = model_sources.get(src, {}) if isinstance(model_sources.get(src), dict) else {}
        for key in ("failures", "sensitive_hits", "attack_requests", "scanner_requests", "client_errors", "refused"):
            if _int(got.get(key)) != _int(vals.get(key)):
                out.append(f"{src}.{key}: model {got.get(key)!r} vs parser {vals.get(key)!r}")
    model_app = model_facts.get("app") if isinstance(model_facts.get("app"), dict) else {}
    for key in ("auth_failures", "attack_reports"):
        if _int(model_app.get(key)) != _int(true_facts.get("app", {}).get(key)):
            out.append(f"app.{key}: model {model_app.get(key)!r} vs parser {true_facts['app'][key]!r}")
    for key in ("commands", "process_chains", "privilege_changes", "persistence"):
        got = model_facts.get(key)
        if len(got if isinstance(got, list) else []) != len(true_facts.get(key, [])):
            out.append(f"{key}: model {len(got) if isinstance(got, list) else 0} vs parser {len(true_facts.get(key, []))}")
    return out


def _ssh_attempt(fields: dict) -> tuple[str, str, str]:
    return str(fields.get("src_ip", "?")), str(fields.get("username", "")).lower(), str(fields.get("port", "") or "")


def _also_failed(attempt: tuple[str, str, str], failed: list[tuple[str, str, str]]) -> bool:
    """Whether a failed line records the same attempt; ports are compared
    only when both lines have one."""
    ip, user, port = attempt
    return any(ip == f_ip and user == f_user and (not port or not f_port or port == f_port)
               for f_ip, f_user, f_port in failed)


def _int(x) -> int:
    """A count from extracted facts: a list counts its items, and a numeric
    string such as "9.0" is read as a number."""
    if isinstance(x, list):
        return len(x)
    try:
        return int(float(x)) if isinstance(x, str) else int(x)
    except (TypeError, ValueError, OverflowError):
        return 0


def _list(x) -> list:
    return [i for i in x if isinstance(i, dict)] if isinstance(x, list) else []
