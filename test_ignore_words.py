"""Tests for backfill ignore-word pattern matching (fnmatch against DiarizationConfig.backfill_ignore_words)."""

import fnmatch

import pytest

from diarize.config import DiarizationConfig


def _is_filtered(word: str, ignore_words) -> bool:
    """Mirror BackfillTranscriber._should_filter_word's matching logic."""
    cleaned = word.strip().lower().strip('.,!?"\'-')

    if cleaned in ignore_words:
        return True

    for pattern in ignore_words:
        if '*' in pattern and fnmatch.fnmatch(cleaned, pattern):
            return True

    return False


@pytest.fixture
def ignore_words():
    # Pass the list explicitly rather than relying on DiarizationConfig's
    # SWHISPER_BACKFILL_IGNORE_WORDS env fallback: CI has no .swhisper.env,
    # so that fallback silently resolves to [] there.
    return DiarizationConfig(backfill_ignore_words=['balans', 'apartheid', 'musik:', 'tack*']).backfill_ignore_words


@pytest.mark.parametrize(
    "word, expected_filtered",
    [
        ('Balans', True),          # exact match, case-insensitive
        ('balans', True),
        ('Apartheid', True),
        ('Musik:', True),          # exact match including the colon
        ('musik', False),          # no colon: not an exact match, no wildcard covers it
        ('Tack', True),            # matches the 'tack*' wildcard
        ('Tackar', True),          # matches the 'tack*' wildcard
        ('Something:', False),     # not in the list, no pattern covers it
        ('test', False),
        ('Hej:', False),
    ],
)
def test_ignore_word_matching(ignore_words, word, expected_filtered):
    assert _is_filtered(word, ignore_words) == expected_filtered
