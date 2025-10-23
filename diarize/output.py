"""Output formatters for various transcript formats."""

import os
import json
import csv
from typing import Dict, List, Any
from .utils import WordProcessor


def format_vtt_timestamp(seconds: float) -> str:
    """Formats time in seconds to VTT format HH:MM:SS.mmm."""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class TranscriptFormatter:
    """Base class for transcript formatters."""
    
    def __init__(self):
        pass
    
    def format(self, segments: List[Dict], output_path: str, **kwargs) -> None:
        """Format segments and save to output path."""
        raise NotImplementedError


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


class RTFFormatter(TranscriptFormatter):
    """Formats transcripts in RTF format with speaker colors."""
    
    def format(self, segments: List[Dict], output_path: str, config=None, transcript_id: str = None, **kwargs) -> None:
        """
        Create an RTF transcript with speaker paragraphs.
        Silences are embedded within the text as word-level markers.
        """
        # RTF header with proper Unicode support
        rtf_content = [
            r"{\rtf1\ansi\ansicpg1252\deff0",
            r"{\fonttbl{\f0 Times New Roman;}}",
            r"""{\colortbl ;\red0\green0\blue0;\red255\green0\blue0;\red0\green0\blue255;\red0\green128\blue0;\red128\green0\blue128;\red255\green165\blue0;\red139\green69\blue19;\red255\green192\blue203;\red0\green128\blue128;\red128\green128\blue0;\red0\green0\blue128;\red128\green0\blue0;\red0\green100\blue0;\red255\green140\blue0;\red64\green224\blue208;\red238\green130\blue238;\red211\green211\blue211;\red105\green105\blue105;\red255\green215\blue0;\red135\green206\blue250;\red250\green128\blue114;\red128\green128\blue128;}""",
            r"\f0\fs24",
            ""
        ]
        
        def escape_rtf_text(text: str) -> str:
            """Escape special characters for RTF and handle Unicode."""
            if not text:
                return ""
            
            # Replace RTF special characters
            text = text.replace('\\', '\\\\')
            text = text.replace('{', r'\{')
            text = text.replace('}', r'\}')
            
            # Handle newlines - convert to RTF paragraph breaks
            text = text.replace('\n', '\\par ')
            
            # Handle Unicode characters
            result = ""
            for char in text:
                if ord(char) > 127:
                    # Convert Unicode to RTF Unicode escape
                    result += f"\\u{ord(char)}?"
                else:
                    result += char
            
            return result
        
        paragraphs = []
        current_speaker = None
        current_paragraph = []
        current_start_time = None
        
        gap_threshold = getattr(config, 'silence_gap_linebreak_threshold', None) if config else None
        
        def smart_join(parts):
            """Join text parts intelligently, preserving newlines around long silences."""
            if not parts:
                return ""
            
            result = parts[0]
            for i in range(1, len(parts)):
                prev_part = parts[i-1]
                curr_part = parts[i]
                
                # If either part contains newlines (indicating a long silence),
                # don't add a space between them
                if '\n' in prev_part or '\n' in curr_part:
                    result += curr_part
                else:
                    result += ' ' + curr_part
            
            return result
        
        for segment in segments:
            is_silence = segment.get('is_silence_marker', False)
            speaker = segment.get('speaker', 'UNKNOWN')

            text = WordProcessor.create_paragraph_text_from_words(segment, gap_threshold=gap_threshold)
            
            if not text:
                continue

            if is_silence:
                # For silence segments, check if it's a long silence and format accordingly
                if current_paragraph:
                    # Parse the duration from the silence text (e.g., "(1.5s)" -> 1.5)
                    duration = WordProcessor._parse_silence_duration(text)
                    if duration is not None and gap_threshold and duration >= gap_threshold:
                        # Long silence - add with blank lines (two newlines create empty lines)
                        formatted_text = f"\n\n{text}\n\n"
                    else:
                        # Short silence - keep inline
                        formatted_text = text
                    current_paragraph.append(formatted_text)
                continue
            
            if speaker != current_speaker:
                # Save the previous paragraph if it exists
                if current_paragraph and current_speaker:
                    paragraph_text = smart_join(current_paragraph)
                    paragraphs.append({
                        'speaker': current_speaker,
                        'text': paragraph_text,
                        'start_time': current_start_time
                    })
                
                # Start new paragraph
                current_speaker = speaker
                current_paragraph = [text] if text else []
                current_start_time = segment.get('start', 0)
            else:
                # Continue current paragraph
                if text:
                    current_paragraph.append(text)
        
        # Don't forget the last paragraph
        if current_paragraph and current_speaker:
            paragraph_text = smart_join(current_paragraph)
            paragraphs.append({
                'speaker': current_speaker,
                'text': paragraph_text,
                'start_time': current_start_time
            })
        
        # Define speaker colors (RTF color indices)
        speaker_colors = {
            'SPEAKER_00': r'\cf2',   # Red
            'SPEAKER_01': r'\cf3',   # Blue
            'SPEAKER_02': r'\cf4',   # Green
            'SPEAKER_03': r'\cf5',   # Purple
            'SPEAKER_04': r'\cf6',   # Orange
            'SPEAKER_05': r'\cf7',   # Brown
            'SPEAKER_06': r'\cf8',   # Pink
            'SPEAKER_07': r'\cf9',   # Teal
            'SPEAKER_08': r'\cf10',  # Olive
            'SPEAKER_09': r'\cf11',  # Navy
            'SPEAKER_10': r'\cf12',  # Maroon
            'SPEAKER_11': r'\cf13',  # Dark Green
            'SPEAKER_12': r'\cf14',  # Dark Orange
            'SPEAKER_13': r'\cf15',  # Turquoise
            'SPEAKER_14': r'\cf16',  # Violet
            'SPEAKER_15': r'\cf17',  # Light Grey
            'SPEAKER_16': r'\cf18',  # Dark Grey
            'SPEAKER_17': r'\cf19',  # Gold
            'SPEAKER_18': r'\cf20',  # Sky Blue
            'SPEAKER_19': r'\cf21',  # Salmon
            'SILENCE':    r'\cf22',  # Grey for silence markers
            'UNKNOWN':    r'\cf1',   # Black (default)
        }

        
        # Add paragraphs to RTF
        # First, add preamble if config is provided and has output_preamble
        if config and hasattr(config, 'output_preamble') and config.output_preamble:
            # Add preamble as a separate section
            if transcript_id and hasattr(config, 'get_preamble_with_transcript_id'):
                preamble_text = config.get_preamble_with_transcript_id(transcript_id)
            else:
                preamble_text = config.output_preamble
            
            escaped_preamble = escape_rtf_text(preamble_text)
            rtf_content.append(f"{escaped_preamble}\\par")
            rtf_content.append("\\par")  # Extra line break
            # Add separator line
            rtf_content.append("\\line " + "=" * 80 + "\\par")
            rtf_content.append("\\par")  # Extra line break after separator
        
        for i, para in enumerate(paragraphs):
            speaker = para['speaker']
            text = escape_rtf_text(para['text'])
            start_time = para['start_time']
            
            # Format timestamp
            hours = int(start_time // 3600)
            minutes = int((start_time % 3600) // 60)
            seconds = int(start_time % 60)
            timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
            
            # Get speaker color
            color = speaker_colors.get(speaker, r'\cf1')
            
            # Add speaker header with timestamp
            rtf_content.append(f"{color}\\b {speaker} {timestamp}:\\b0\\cf1\\par")
            
            # Add paragraph text
            rtf_content.append(f"{text}\\par")
            rtf_content.append("\\par")  # Extra line break between speakers
        
        # RTF footer
        rtf_content.append("}")
        
        # Write RTF file with proper encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(rtf_content))


class TXTFormatter(TranscriptFormatter):
    """Formats transcripts in plain text format."""
    
    def format(self, segments: List[Dict], output_path: str, include_silence: bool = True, config=None, transcript_id: str = None, **kwargs) -> None:
        """
        Create a plain text transcript with speaker paragraphs.
        Adjacent segments from the same speaker are joined together.
        Depending on settings, blank lines may be added around long silence (even when there is no speaker change).
        """
        # Group segments by speaker (but handle silence markers separately)
        gap_threshold = getattr(config, 'silence_gap_linebreak_threshold', None) if config else None
        
        paragraphs = []
        current_speaker = None
        current_paragraph = []
        current_start_time = None
        
        def smart_join(parts):
            """Join text parts intelligently, preserving newlines around long silences."""
            if not parts:
                return ""
            
            result = parts[0]
            for i in range(1, len(parts)):
                prev_part = parts[i-1]
                curr_part = parts[i]
                
                # If either part contains newlines (indicating a long silence),
                # don't add a space between them
                if '\n' in prev_part or '\n' in curr_part:
                    result += curr_part
                else:
                    result += ' ' + curr_part
            
            return result

        for segment in segments:
            is_silence = segment.get('is_silence_marker', False)
            speaker = segment.get('speaker', 'UNKNOWN')

            text = WordProcessor.create_paragraph_text_from_words(segment, gap_threshold=gap_threshold)
            
            if not text:
                continue

            if is_silence:
                # For silence segments, check if it's a long silence and format accordingly
                if current_paragraph:
                    # Parse the duration from the silence text (e.g., "(1.5s)" -> 1.5)
                    duration = WordProcessor._parse_silence_duration(text)
                    if duration is not None and gap_threshold and duration >= gap_threshold:
                        # Long silence - add with blank lines (two newlines create empty lines)
                        formatted_text = f"\n\n{text}\n\n"
                    else:
                        # Short silence - keep inline
                        formatted_text = text
                    current_paragraph.append(formatted_text)
                continue
            
            text = WordProcessor.create_paragraph_text_from_words(segment, gap_threshold=gap_threshold)
            
            if speaker != current_speaker:
                # Save the previous paragraph if it exists
                if current_paragraph and current_speaker:
                    paragraph_text = smart_join(current_paragraph)
                    paragraphs.append({
                        'speaker': current_speaker,
                        'text': paragraph_text,
                        'start_time': current_start_time,
                        'is_silence': False
                    })
                
                # Start new paragraph
                current_speaker = speaker
                current_paragraph = [text] if text else []
                current_start_time = segment.get('start', 0)
            else:
                # Continue current paragraph
                if text:
                    current_paragraph.append(text)
        
        # Don't forget the last paragraph
        if current_paragraph and current_speaker:
            paragraph_text = smart_join(current_paragraph)
            paragraphs.append({
                'speaker': current_speaker,
                'text': paragraph_text,
                'start_time': current_start_time,
                'is_silence': False
            })
        
        # Create text content
        txt_content = []
        
        # Add preamble if config is provided and has output_preamble
        if config and hasattr(config, 'output_preamble') and config.output_preamble:
            if transcript_id and hasattr(config, 'get_preamble_with_transcript_id'):
                preamble_text = config.get_preamble_with_transcript_id(transcript_id)
            else:
                preamble_text = config.output_preamble
            txt_content.append(preamble_text)
            txt_content.append("")        # Empty line after preamble
            txt_content.append("=" * 80)  # Separator line
            txt_content.append("")        # Empty line after separator
        
        for para in paragraphs:
            speaker = para['speaker']
            text = para['text']
            start_time = para['start_time']
            is_silence = para.get('is_silence', False)
            
            if not is_silence:
                # Format timestamp
                hours = int(start_time // 3600)
                minutes = int((start_time % 3600) // 60)
                seconds = int(start_time % 60)
                timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
                
                # Add speaker header with timestamp
                txt_content.append(f"{speaker} {timestamp}:")
                txt_content.append(text)
                txt_content.append("")  # Empty line between speakers
        
        # Write text file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt_content))


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
