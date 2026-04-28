"""Defensive sanitization for secret env vars that end up in HTTP headers/URLs.

Background — 2026-04-28 production outage:
The scanner's `x-api-key: <ANTHROPIC_API_KEY>` header crashed with
`'ascii' codec can't encode character '\\u2028' in position 108`
on every Anthropic call (Haiku, Sonnet, Opus) for ~2 weeks. The
ANTHROPIC_API_KEY env var on Railway had a U+2028 LINE SEPARATOR baked
in — invisible in any UI, presumably from a copy-paste off a doc/PDF
that converted a newline to U+2028 instead of \\n.

httpx encodes header values as ASCII/latin-1 — any character outside
that range detonates request construction. The same risk applies to
OANDA tokens (Authorization header), Telegram bot tokens (URL path),
and ADMIN_SECRET (compared to header). Every secret read from the
environment that ends up on the wire must pass through here at load
time.

This is intentionally a tiny, dependency-free module so it can be
imported cheaply from anywhere without circular-import risk.
"""

# Characters that are invisible in editors/UIs but break ASCII/latin-1
# encoding of HTTP headers. Whitespace .strip() does NOT remove these.
_INVISIBLE_CHARS = (
    "\u2028",  # LINE SEPARATOR — the one that hit us in production
    "\u2029",  # PARAGRAPH SEPARATOR — same family, same risk
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
    "\r",      # carriage return — would also break headers
    "\n",      # newline — would also break headers
)


def sanitize_env_secret(value):
    """Return a header-safe version of a secret pulled from os.environ.

    Strips leading/trailing whitespace, then removes any embedded
    invisible Unicode separators / zero-width chars / CR / LF that
    could survive a copy-paste from a rich-text source.

    Returns "" for None or empty input — callers that need a "missing"
    signal should check the truthiness of the result, matching the
    behaviour of `os.environ.get(KEY, "")`.

    Args:
        value: raw env var value, typically from `os.environ.get(...)`.

    Returns:
        Cleaned string safe for use in HTTP headers and URLs.
    """
    if not value:
        return ""
    cleaned = value.strip()
    for ch in _INVISIBLE_CHARS:
        if ch in cleaned:
            cleaned = cleaned.replace(ch, "")
    return cleaned
