"""Tests for env_utils.sanitize_env_secret.

Background — 2026-04-28 Railway outage:
The scanner's `x-api-key` header was crashing with
`'ascii' codec can't encode character '\\u2028' in position 108`
on every Anthropic call (Haiku, Sonnet, Opus). Root cause: an invisible
U+2028 LINE SEPARATOR baked into the ANTHROPIC_API_KEY env var on Railway
(copied from a doc/PDF that converted a newline to U+2028). httpx
encodes header values as ASCII/latin-1 — U+2028 detonates.

Defensive fix: every secret env var pulled into a header or URL must be
passed through sanitize_env_secret() at load time.
"""
import pytest

from ml.env_utils import sanitize_env_secret


class TestSanitizeEnvSecret:
    def test_clean_key_unchanged(self):
        key = "sk-ant-api03-AbCdEf-0123456789"
        assert sanitize_env_secret(key) == key

    def test_strips_leading_trailing_whitespace(self):
        assert sanitize_env_secret("  sk-ant-key  ") == "sk-ant-key"
        assert sanitize_env_secret("\tsk-ant-key\t") == "sk-ant-key"

    def test_strips_trailing_newline(self):
        # Common when copying from a doc that has a trailing line break
        assert sanitize_env_secret("sk-ant-key\n") == "sk-ant-key"
        assert sanitize_env_secret("sk-ant-key\r\n") == "sk-ant-key"

    def test_removes_unicode_line_separator_u2028(self):
        # The actual bug from production
        assert sanitize_env_secret("sk-ant-key\u2028") == "sk-ant-key"
        assert sanitize_env_secret("sk-ant\u2028key") == "sk-antkey"

    def test_removes_unicode_paragraph_separator_u2029(self):
        assert sanitize_env_secret("sk-ant-key\u2029") == "sk-ant-key"
        assert sanitize_env_secret("sk\u2029ant\u2029key") == "skantkey"

    def test_removes_zero_width_chars(self):
        # U+200B ZERO WIDTH SPACE, U+200C ZWNJ, U+200D ZWJ, U+FEFF BOM
        assert sanitize_env_secret("\ufeffsk-ant-key") == "sk-ant-key"
        assert sanitize_env_secret("sk\u200bant\u200ckey") == "skantkey"
        assert sanitize_env_secret("sk\u200dant-key") == "skant-key"

    def test_removes_embedded_cr_lf(self):
        assert sanitize_env_secret("sk-ant\nkey") == "sk-antkey"
        assert sanitize_env_secret("sk-ant\rkey") == "sk-antkey"

    def test_empty_string_returns_empty(self):
        assert sanitize_env_secret("") == ""

    def test_none_returns_empty(self):
        # Defensive — callers may pass os.environ.get() result with no default
        assert sanitize_env_secret(None) == ""

    def test_resulting_string_is_ascii_encodable(self):
        # The whole point: result must round-trip through ASCII (HTTP header path)
        bad = "sk-ant-api03-AbCd\u2028\u200bEfGh\n  "
        cleaned = sanitize_env_secret(bad)
        # Should not raise
        cleaned.encode("ascii")

    def test_preserves_non_secret_punctuation(self):
        # Hyphens, underscores, dots — keys legitimately contain these
        assert sanitize_env_secret("sk-ant_api.03-key") == "sk-ant_api.03-key"
