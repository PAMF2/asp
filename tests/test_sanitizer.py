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

    def test_metadata_contains_only_safe_fields(self):
        """Metadata must not contain any prompt-derived features."""
        sanitizer = SanitizerGateway()
        result = sanitizer.sanitize("one two three four five")
        # Safe processing metadata should be present
        assert "processed_at" in result.metadata
        assert "asp_version" in result.metadata
        # Prompt-derived features must NOT be present
        assert "token_count" not in result.metadata
        assert "has_code_blocks" not in result.metadata
        assert "language_hint" not in result.metadata

    def test_metadata_no_prompt_leakage_across_inputs(self):
        """Different prompts must produce identical metadata keys."""
        sanitizer = SanitizerGateway()
        a = sanitizer.sanitize("short")
        b = sanitizer.sanitize("a much longer prompt with many words " * 10)
        c = sanitizer.sanitize("```python\nprint('hi')```")
        # All should have the same metadata keys (no input-dependent fields)
        assert set(a.metadata.keys()) == set(b.metadata.keys()) == set(c.metadata.keys())

    def test_custom_preamble(self):
        sanitizer = SanitizerGateway(preamble="Custom preamble here")
        result = sanitizer.sanitize("test")
        assert result.alignment_preamble == "Custom preamble here"
