"""Tests for the diarization-first pipeline's config and speaker-chunking logic.

Pure logic only: no audio files or model loading.
"""

from transcribe.config import TranscriptionConfig
from transcribe.speaker_chunking import (
    analyze_chunk_distribution,
    convert_segments_to_vad_format,
    create_speaker_aware_chunks,
    get_speaker_for_timestamp,
)


def test_config_exposes_diarization_first_settings():
    config = TranscriptionConfig()

    assert hasattr(config, 'diarization_first')
    assert hasattr(config, 'use_speaker_segments_as_vad')
    assert hasattr(config, 'speaker_aware_chunking')
    assert hasattr(config, 'max_chunk_duration')
    assert hasattr(config, 'min_chunk_duration')


SPEAKER_SEGMENTS = [
    {'start': 0.0, 'end': 45.2, 'speaker': 'SPEAKER_00'},
    {'start': 45.2, 'end': 92.8, 'speaker': 'SPEAKER_01'},
    {'start': 92.8, 'end': 150.3, 'speaker': 'SPEAKER_00'},
    {'start': 150.3, 'end': 185.7, 'speaker': 'SPEAKER_01'},
    {'start': 185.7, 'end': 220.4, 'speaker': 'SPEAKER_00'},
    {'start': 220.4, 'end': 280.5, 'speaker': 'SPEAKER_01'},
]


def test_speaker_aware_chunks_cover_all_segments():
    chunks = create_speaker_aware_chunks(
        SPEAKER_SEGMENTS,
        target_duration=180.0,
        max_duration=200.0,
        min_chunk_duration=30.0,
    )

    assert len(chunks) > 0
    total_segments_in_chunks = sum(len(chunk['segments']) for chunk in chunks)
    assert total_segments_in_chunks == len(SPEAKER_SEGMENTS)
    for chunk in chunks:
        assert chunk['chunk_end'] > chunk['chunk_start']


def test_convert_segments_to_vad_format_merges_contiguous_segments():
    # SPEAKER_SEGMENTS has zero gap between consecutive turns, which is below
    # the default 0.3s merge_gap_threshold, so a whole chunk collapses into a
    # single VAD span rather than one tuple per input segment.
    chunks = create_speaker_aware_chunks(
        SPEAKER_SEGMENTS,
        target_duration=180.0,
        max_duration=200.0,
        min_chunk_duration=30.0,
    )
    first_chunk = chunks[0]

    vad_segments = convert_segments_to_vad_format(first_chunk['segments'], first_chunk['chunk_start'])

    assert vad_segments == [(0.0, first_chunk['chunk_end'] - first_chunk['chunk_start'])]


def test_convert_segments_to_vad_format_keeps_large_gaps_separate():
    segments = [
        {'start': 10.0, 'end': 15.0, 'speaker': 'SPEAKER_00'},
        {'start': 16.0, 'end': 20.0, 'speaker': 'SPEAKER_01'},  # 1.0s gap, above default 0.3s threshold
    ]

    vad_segments = convert_segments_to_vad_format(segments, chunk_start=10.0)

    assert vad_segments == [(0.0, 5.0), (6.0, 10.0)]


def test_get_speaker_for_timestamp():
    assert get_speaker_for_timestamp(100.0, SPEAKER_SEGMENTS) == 'SPEAKER_00'
    assert get_speaker_for_timestamp(60.0, SPEAKER_SEGMENTS) == 'SPEAKER_01'


def test_analyze_chunk_distribution():
    chunks = create_speaker_aware_chunks(
        SPEAKER_SEGMENTS,
        target_duration=180.0,
        max_duration=200.0,
        min_chunk_duration=30.0,
    )

    stats = analyze_chunk_distribution(chunks)

    assert stats['total_chunks'] == len(chunks)
    assert stats['total_duration'] > 0
    assert stats['min_chunk_duration'] <= stats['avg_chunk_duration'] <= stats['max_chunk_duration']
