"""Tests for transcript_render.formatters.smart_join.

smart_join used to be copy-pasted independently inside RTFFormatter.format
and TXTFormatter.render; this test exercises the single shared
implementation both call sites now use.
"""

from transcript_render.formatters import smart_join


def test_empty_input_returns_empty_string():
    assert smart_join([]) == ""


def test_single_part_returned_unchanged():
    assert smart_join(["hello"]) == "hello"


def test_inserts_space_between_plain_parts():
    assert smart_join(["hello", "world"]) == "hello world"


def test_no_extra_space_after_trailing_newline():
    assert smart_join(["Ja\n\n(5.1)\n\nja", "another segment"]) == "Ja\n\n(5.1)\n\nja another segment"


def test_no_extra_space_before_leading_whitespace():
    assert smart_join(["hello", " world"]) == "hello world"
