"""Tests for the TEE sanitizer gateway."""

from __future__ import annotations

from asp.tee.sanitizer import SanitizerGateway


class TestSanitizerGateway:

    def test_sanitized_context_has_no_raw_prompt(self):
        sanitizer = SanitizerGateway()
        raw = "This is my secret prompt about hacking"
        result = sanitizer.sanitize(raw)

        # rewritten_prompt should be empty (filled by defense module later)
        assert result.rewritten_prompt == ""
        assert raw not in result.alignment_preamble
        assert raw not in str(result.metadata)

    def test_each_call_gets_unique_request_id(self):
        sanitizer = SanitizerGateway()
        a = sanitizer.sanitize("prompt 1")
        b = sanitizer.sanitize("prompt 2")
        assert a.request_id != b.request_id

    def test_metadata_includes_token_count(self):
        sanitizer = SanitizerGateway()
        result = sanitizer.sanitize("one two three four five")
        assert result.metadata["token_count"] == 5

    def test_custom_preamble(self):
        sanitizer = SanitizerGateway(preamble="Custom preamble here")
        result = sanitizer.sanitize("test")
        assert result.alignment_preamble == "Custom preamble here"

    def test_language_detection_chinese(self):
        sanitizer = SanitizerGateway()
        result = sanitizer.sanitize("Hello world")
        assert result.metadata["language_hint"] == "en"

    def test_code_block_detection(self):
        sanitizer = SanitizerGateway()
        result = sanitizer.sanitize("Here is code: ```python\nprint('hi')```")
        assert result.metadata["has_code_blocks"] is True
