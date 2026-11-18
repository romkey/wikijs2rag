"""Deterministic sanitization, run on every page before it can be indexed.

1. Secret scanner: well-known credential formats (in the spirit of gitleaks'
   rules) plus high-entropy values assigned to secret-sounding keys. A hit is
   redacted in place and reported; the secret itself is never stored.
2. Network identifiers in configs: IP and MAC addresses, internal hostnames
   and Home Assistant entity IDs inside code blocks become placeholders.

The LLM exclusion classifier lives in enrich.py; it isn't deterministic.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

REDACTED = "[REDACTED]"

# (rule name, pattern). The whole match is redacted unless the pattern has a
# group named "secret", in which case only that group is.
_SECRET_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "private-key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----.*?-----END [A-Z ]*PRIVATE KEY(?: BLOCK)?-----", re.S
        ),
    ),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{60,}\b")),
    ("gitlab-token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("slack-webhook", re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+")),
    ("discord-webhook", re.compile(r"https://(?:ptb\.|canary\.)?discord(?:app)?\.com/api/webhooks/\d+/[\w-]+")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("stripe-key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}\b")),
    ("openai-key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{32,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("url-credentials", re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:(?P<secret>[^\s@/]{3,})@")),
    (
        "assigned-secret",
        re.compile(
            r"(?i)\b[\w.-]*(?:password|passwd|pwd|passphrase|secret|api[_-]?key|token|auth[_-]?key|"
            r"encryption[_-]?key|ota[_-]?password|wifi[_-]?password|psk)[\w.-]*"
            r"[\"']?\s*[:=]\s*[\"']?(?P<secret>[^\s\"',;]{6,})"
        ),
    ),
]

# Values that look like placeholders or references, not secrets.
_PLACEHOLDER = re.compile(
    r"(?i)^(?:\*+|x+|\.+|<[^>]*>|\$\{?[\w.]+\}?|!secret\b.*|\{\{.*\}\}|changeme|change_me|replace_?me|"
    r"your[_-].*|example.*|placeholder|none|null|true|false|redacted|\[redacted\])$"
)
_ENTROPY_MIN = 2.5  # bits per character, for assigned-secret values


def _entropy(value: str) -> float:
    counts = Counter(value)
    return -sum(n / len(value) * math.log2(n / len(value)) for n in counts.values())


@dataclass
class SecretHit:
    rule: str
    line: int  # 1-based line number in the scanned text


def scan_secrets(text: str) -> tuple[str, list[SecretHit]]:
    """Return *text* with secrets redacted, and where they were."""
    hits: list[SecretHit] = []
    spans: list[tuple[int, int, str]] = []
    for rule, pattern in _SECRET_RULES:
        for m in pattern.finditer(text):
            start, end = m.span("secret") if "secret" in pattern.groupindex else m.span()
            value = text[start:end]
            if rule == "assigned-secret" and (_PLACEHOLDER.match(value) or _entropy(value) < _ENTROPY_MIN):
                continue
            if any(s < end and start < e for s, e, _ in spans):
                continue
            spans.append((start, end, rule))
    for start, end, rule in sorted(spans, reverse=True):
        hits.append(SecretHit(rule, text.count("\n", 0, start) + 1))
        text = text[:start] + REDACTED + text[end:]
    hits.reverse()
    return text, hits


_IPV4 = re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?:/\d{1,2})?\b")
_IPV6 = re.compile(
    r"(?<![\w:])(?:(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}|(?:[0-9A-Fa-f]{1,4}:)*:(?::?[0-9A-Fa-f]{1,4})+)(?![\w:])"
)
_MAC = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")
_HOSTNAME = re.compile(r"\b[a-zA-Z0-9][\w-]*(?:\.[\w-]+)*\.(?:local|lan|internal|home\.arpa|localdomain)\b")
_ENTITY_ID = re.compile(
    r"\b(?:sensor|binary_sensor|switch|light|lock|cover|climate|camera|fan|input_boolean|input_number|"
    r"input_select|automation|script|media_player|alarm_control_panel|button|number|select|device_tracker)"
    r"\.[a-z0-9_]+\b"
)
_CODE_BLOCK = re.compile(r"^(\s*)(```+|~~~+)[^\n]*\n.*?^\s*\2\s*$", re.M | re.S)


def _placeholders(code: str) -> str:
    code = _MAC.sub("<MAC>", code)
    code = _IPV6.sub("<IP>", code)
    code = _IPV4.sub("<IP>", code)
    code = _HOSTNAME.sub("<HOSTNAME>", code)
    return _ENTITY_ID.sub("<ENTITY_ID>", code)


def scrub_network_identifiers(markdown: str) -> str:
    """Replace network identifiers inside fenced code blocks with placeholders."""
    return _CODE_BLOCK.sub(lambda m: _placeholders(m.group(0)), markdown)
