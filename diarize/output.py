"""Output formatters for various transcript formats."""

import os
import json
import csv
from typing import Any, Dict, List, Optional
from .utils import WordProcessor

# The plain-text and RTF formatters live in ``transcript_render`` so they can
# run without importing the diarization stack. Re-exported here because
# ``diarize.output`` is the historic import site.
from transcript_render.formatters import (
    TranscriptFormatter,
    TXTFormatter,
    RTFFormatter,
    SPEAKER_STATS_WARNING,
)


def format_vtt_timestamp(seconds: float) -> str:
    """Formats time in seconds to VTT format HH:MM:SS.mmm."""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class VTTFormatter(TranscriptFormatter):
    """Formats transcripts in VTT format."""
    
    def format(self, segments: List[Dict], output_path: str, **kwargs) -> None:
        """Save the transcript in VTT format."""
        vtt_content = ["WEBVTT", ""]
        
        for segment in segments:
            # Skip standalone silence segments
            if segment.get('is_silence_marker', False):
                continue
            
            start_time = format_vtt_timestamp(segment['start'])
            end_time = format_vtt_timestamp(segment['end'])
            speaker = segment.get('speaker', 'UNKNOWN')
            
            # Reconstruct segments from words to preserve ALL markers (including silences)
            words = segment.get('words', [])
            if words:
                word_texts = []
                for word in words:
                    word_text = word.get('word', word.get('text', ''))
                    if word_text:  # Include [*] markers AND (1.2) silence markers
                        word_texts.append(word_text)
                text = ' '.join(word_texts).strip()
            else:
                text = segment.get('text', '').strip()  # Fallback

            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(f"<{speaker}> {text}")
            vtt_content.append("")
        
        with open(output_path, "w", encoding='utf-8') as f:
            f.write("\n".join(vtt_content))


class RTTMFormatter(TranscriptFormatter):
    """Formats transcripts in RTTM format."""
    
    def format(self, segments: List[Dict], output_path: str, audio_basename: str = "", **kwargs) -> int:
        """
        Create an RTTM file from diarized segments.
        
        RTTM format: SPEAKER <filename> <channel> <start> <duration> <ortho> <stype> <name> <conf> <slat>
        """
        rttm_lines = []
        
        # Get base filename without extension for RTTM
        base_name = os.path.splitext(audio_basename)[0] if audio_basename else "audio"
        
        for segment in segments:
            speaker = segment.get('speaker', 'UNKNOWN')
            start_time = segment['start']
            duration = segment['end'] - segment['start']
            confidence = segment.get('speaker_confidence', 0.0)
            
            # Skip silence markers in RTTM (they're not speakers)
            if speaker == 'SILENCE':
                continue
            
            # RTTM format line
            rttm_line = f"SPEAKER {base_name} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker} {confidence:.3f} <NA>"
            rttm_lines.append(rttm_line)
        
        # Write RTTM file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(rttm_lines))
        
        return len(rttm_lines)
    
    def format_detailed(self, segments: List[Dict], output_path: str, audio_basename: str = "", **kwargs) -> int:
        """
        Create a detailed RTTM file with word-level timing information.
        This creates one RTTM entry per word, showing fine-grained speaker assignments.
        """
        rttm_lines = []
        
        # Get base filename without extension for RTTM
        base_name = os.path.splitext(audio_basename)[0] if audio_basename else "audio"
        
        for segment in segments:
            words = segment.get('words', [])
            
            for word in words:
                speaker = word.get('speaker', 'UNKNOWN')
                start_time = word.get('start', segment['start'])
                end_time = word.get('end', segment['end'])
                duration = end_time - start_time
                confidence = word.get('confidence', 0.0)
                
                # Skip silence markers and very short words
                if speaker == 'SILENCE' or duration < 0.01:
                    continue
                
                # RTTM format line for word-level
                rttm_line = f"SPEAKER {base_name} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker} {confidence:.3f} <NA>"
                rttm_lines.append(rttm_line)
        
        # Write detailed RTTM file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(rttm_lines))
        
        return len(rttm_lines)


class TSVFormatter(TranscriptFormatter):
    """Formats transcripts in TSV format."""

    def format(
        self,
        segments: List[Dict],
        output_path: str,
        include_silence: bool = True,
        include_word_details: bool = True,
        config=None,
        word_per_line: bool = False,  # DO NOT CHANGE, this is just the default. Change DiarizationConfig.tsv_word_per_line instead
        **kwargs
    ) -> None:
        """Create a TSV transcript with optional word-level metadata."""

        fieldnames = [
            'segment_index',
            'start',
            'end',
            'duration',
            'speaker',
            'speaker_confidence',
            'is_silence',
            'text',
            'word_count'
        ]

        if include_word_details and not word_per_line:
            fieldnames.append('words')

        if word_per_line:
            fieldnames.extend([
                'word',
                'word_start',
                'word_end',
                'word_speaker',
                'word_confidence',
                'word_is_silence'
            ])

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()

            for index, segment in enumerate(segments):
                is_silence_marker = segment.get('is_silence_marker', False)
                if is_silence_marker and not include_silence:
                    continue

                start_time = float(segment.get('start', 0) or 0)
                end_time = float(segment.get('end', 0) or 0)
                duration = max(end_time - start_time, 0.0)
                speaker = segment.get('speaker', 'UNKNOWN')
                speaker_confidence = segment.get('speaker_confidence')
                text = WordProcessor.create_paragraph_text_from_words(segment, gap_threshold=None)
                if not text:
                    text = segment.get('text', '').strip()

                if text:
                    # Collapse blank lines created for silence markers and keep markers inline
                    text_lines = [line.strip() for line in text.splitlines() if line.strip()]
                    text = ' '.join(text_lines)
                    text = text.replace('\t', ' ')

                words = segment.get('words', []) or []
                word_count = sum(1 for w in words if not w.get('is_silence_marker', False))

                base_row = {
                    'segment_index': index,
                    'start': round(start_time, 3),
                    'end': round(end_time, 3),
                    'duration': round(duration, 3),
                    'speaker': speaker,
                    'speaker_confidence': round(speaker_confidence, 3) if isinstance(speaker_confidence, (int, float)) else '',
                    'is_silence': bool(is_silence_marker),
                    'text': text,
                    'word_count': word_count
                }

                if word_per_line:
                    for word in words:
                        if word.get('is_silence_marker', False) and not include_silence:
                            continue

                        word_text = word.get('word', word.get('text', '')) or ''
                        word_row = base_row.copy()
                        word_row.update({
                            'word': word_text.replace('\t', ' '),
                            'word_start': round(word.get('start', start_time), 3) if word.get('start') is not None else '',
                            'word_end': round(word.get('end', end_time), 3) if word.get('end') is not None else '',
                            'word_speaker': word.get('speaker', speaker),
                            'word_confidence': round(word.get('confidence'), 3) if isinstance(word.get('confidence'), (int, float)) else '',
                            'word_is_silence': word.get('is_silence_marker', False)
                        })
                        writer.writerow(word_row)

                    if not words:
                        writer.writerow(base_row)

                else:
                    row = dict(base_row)
                    if include_word_details:
                        row['words'] = json.dumps([
                            {
                                'word': word.get('word', word.get('text', '')),
                                'start': word.get('start'),
                                'end': word.get('end'),
                                'speaker': word.get('speaker'),
                                'confidence': word.get('confidence'),
                                'is_silence_marker': word.get('is_silence_marker', False)
                            }
                            for word in words
                        ], ensure_ascii=False).replace('\t', ' ')

                    writer.writerow(row)


class PyannoteSegmentFormatter(TranscriptFormatter):
    """Formats pyannote diarization segments in TSV format."""

    def format(
        self,
        diarization_result,
        output_path: str,
        audio_basename: str = "",
        **kwargs
    ) -> None:
        """Create a TSV file with pyannote segment boundaries."""

        fieldnames = [
            'segment_index',
            'start',
            'end',
            'duration',
            'speaker',
            'confidence'
        ]

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()

            for index, (segment, _, speaker) in enumerate(diarization_result.itertracks(yield_label=True)):
                start_time = float(segment.start)
                end_time = float(segment.end)
                duration = end_time - start_time
                
                # Get confidence if available (some diarization results include this)
                confidence = getattr(segment, 'confidence', None)
                if confidence is None:
                    confidence = ''
                else:
                    confidence = round(float(confidence), 3)

                row = {
                    'segment_index': index,
                    'start': round(start_time, 3),
                    'end': round(end_time, 3),
                    'duration': round(duration, 3),
                    'speaker': speaker,
                    'confidence': confidence
                }

                writer.writerow(row)


class JSONFormatter(TranscriptFormatter):
    """Formats the complete enriched transcript in JSON format."""
    
    def format(
        self,
        segments: List[Dict],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Save the enriched transcript with all metadata to JSON.
        
        Args:
            segments: List of segment dictionaries with words and metadata.
                     Segments may include optional SenseVoice fields:
                     - sensevoice_text: Transcription from SenseVoice
                     - sensevoice_raw_text: Raw text with emotion/event tags
                     - sensevoice_emotion: Detected emotion (e.g., "<|HAPPY|>")
                     - sensevoice_event: Detected event (e.g., "<|Speech|>")
                     - sensevoice_language: Detected language tag
            output_path: Path to save the JSON file
            metadata: Optional metadata about the transcription/diarization process
        """
        output_data = {
            'metadata': metadata or {},
            'segments': segments
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


class StatsExporter:
    """Exports analysis statistics to JSON format."""
    
    @staticmethod
    def save_analysis_stats(segments: List[Dict], diarization_result, word_stats: Dict, 
                          segment_stats: Dict, speaker_stats: Dict, boundary_stats: Dict,
                          settings: Dict, output_path: str) -> None:
        """Save comprehensive analysis statistics to JSON file."""
        
        # Create comprehensive stats dictionary
        analysis_stats = {
            "metadata": {
                "script_version": settings.get("script_version", "unknown"),
                "audio_file": settings.get("audio_file", "unknown"),
                "json_input_file": settings.get("json_input_file", "unknown"),
                "processing_timestamp": settings.get("processing_timestamp", "unknown"),
                "total_segments": len(segments)
            },
            "configuration": {
                "device": settings.get("device", "unknown"),
                "diarization": settings.get("diarization", {}),
                "word_level_processing": settings.get("word_level_processing", {}),
                "output_formats": settings.get("output_formats", [])
            },
            "statistics": {
                "word_level": word_stats,
                "segment_level": segment_stats,
                "speaker_level": speaker_stats,
                "boundary_analysis": boundary_stats
            },
            "segments_summary": {
                "total_segments": len(segments),
                "speakers_detected": len(set(seg.get('speaker', 'UNKNOWN') for seg in segments)),
                "total_duration": max((seg.get('end', 0) for seg in segments), default=0),
                "segments_by_speaker": {}
            }
        }
        
        # Add speaker-specific segment counts
        for segment in segments:
            speaker = segment.get('speaker', 'UNKNOWN')
            if speaker not in analysis_stats["segments_summary"]["segments_by_speaker"]:
                analysis_stats["segments_summary"]["segments_by_speaker"][speaker] = 0
            analysis_stats["segments_summary"]["segments_by_speaker"][speaker] += 1
        
        # Save to JSON file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_stats, f, indent=2, ensure_ascii=False)
