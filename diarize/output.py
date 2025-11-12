"""Output formatters for various transcript formats."""

import os
import re
import json
import csv
from typing import Any, Dict, List, Optional
from .utils import WordProcessor, _is_silence_token

# Warning text about speaker statistics used in both TXT and RTF outputs.
# Keep the human-readable plain text here and an RTF-escaped version for use
# in the RTF generator where backslashes and braces must be escaped.
SPEAKER_STATS_WARNING = (
    "Approximate speaker statistics:\n"
    "(may be very inaccurate, especially if number of detected speakers is incorrect)"
)

# RTF needs backslashes and braces escaped and newlines converted to \par
# We use the plain-text SPEAKER_STATS_WARNING for both TXT and RTF outputs.
# For RTF, escape it at the point of use with `escape_rtf_text`.


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
    
    def format(
        self,
        segments: List[Dict],
        output_path: str,
        config=None,
        transcript_id: str = None,
        speaker_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
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
        
        summary_data = TXTFormatter._get_speaker_summary_table_data(segments, speaker_stats)

        def build_summary_table(table_data: Dict[str, Any]) -> List[str]:
            if not table_data:
                return []

            columns = table_data['columns']
            display_rows = table_data['rows']
            display_gaps = table_data['gap_row']
            display_total = table_data['total_row']
            column_widths = table_data['column_widths']

            # Approximate column width in twips (characters * 160 twips)
            base_twip_per_char = 160
            min_column_twips = 1200
            column_twips: List[int] = []
            for key, _label, _alignment in columns:
                char_width = column_widths.get(key, len(_label))
                width = max(min_column_twips, (char_width + 2) * base_twip_per_char)
                column_twips.append(width)

            cell_edges: List[int] = []
            cumulative = 0
            for width in column_twips:
                cumulative += width
                cell_edges.append(cumulative)

            align_map = {
                'left': r'\ql',
                'right': r'\qr',
                'center': r'\qc',
            }

            table_lines: List[str] = []

            def append_row(cells: List[str], *, bold: bool = False) -> None:
                table_lines.append(r"\trowd\trgaph108")
                for edge in cell_edges:
                    table_lines.append(fr"\cellx{edge}")

                for (cell_value, (_key, _label, alignment)) in zip(cells, columns):
                    alignment_ctrl = align_map.get(alignment, r'\ql')
                    content = escape_rtf_text(cell_value)
                    if bold:
                        content = f"\\b {content}\\b0" if content else r"\b\b0"
                    table_lines.append(rf"\pard\intbl{alignment_ctrl}\sb0\sa0 {content}\cell")

                table_lines.append(r"\row")

            header_cells = [label for _key, label, _alignment in columns]
            append_row(header_cells, bold=True)

            for row in display_rows:
                append_row([row.get(key, '') or '' for key, _label, _alignment in columns])

            if display_gaps:
                append_row([display_gaps.get(key, '') or '' for key, _label, _alignment in columns])

            if display_total:
                append_row([display_total.get(key, '') or '' for key, _label, _alignment in columns], bold=True)

            table_lines.append(r"\pard\par")
            return table_lines

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
                
                # Check if we need a space separator
                # Add space unless the previous part ends with whitespace (including newlines)
                # or the current part starts with whitespace
                needs_space = True
                if prev_part and prev_part[-1] in (' ', '\n', '\t', '\r'):
                    needs_space = False
                if curr_part and curr_part[0] in (' ', '\n', '\t', '\r'):
                    needs_space = False
                
                if needs_space:
                    result += ' ' + curr_part
                else:
                    result += curr_part
            
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
                    # For silence segments, find the first silence marker word to get duration
                    duration = None
                    words = segment.get('words', [])
                    for word in words:
                        if word.get('is_silence_marker', False):
                            duration = WordProcessor._parse_silence_duration(word.get('word', ''))
                            break
                    
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

        
        # Add preamble and optional summary table
        if config and hasattr(config, 'output_preamble') and config.output_preamble:
            if transcript_id and hasattr(config, 'get_preamble_with_transcript_id'):
                preamble_text = config.get_preamble_with_transcript_id(transcript_id)
            else:
                preamble_text = config.output_preamble

            escaped_preamble = escape_rtf_text(preamble_text)
            rtf_content.append(f"{escaped_preamble}\\par")
            rtf_content.append("\\par")

            if summary_data:
                # Bold only the first line (heading) of the speaker statistics warning
                heading, _sep, remainder = SPEAKER_STATS_WARNING.partition("\n")
                rtf_content.append(r"\pard\sb200\sa120\b " + escape_rtf_text(heading) + r"\b0\par")
                if remainder:
                    rtf_content.append(escape_rtf_text(remainder) + r"\par")
                rtf_content.extend(build_summary_table(summary_data))

            # Insert horizontal rule (paragraph border) instead of ASCII equals
            # \\pard starts a new paragraph; \\brdrb specifies a bottom border (single line);
            # \\brdrw10 sets border width; \\brsp sets spacing between border and text.
            rtf_content.append(r"\pard\brdrb\brdrs\brdrw10\brsp20\par")
            rtf_content.append("\\par")
        elif summary_data:
            # Bold only the first line (heading) of the speaker statistics warning
            heading, _sep, remainder = SPEAKER_STATS_WARNING.partition("\n")
            rtf_content.append(r"\pard\sb200\sa120\b " + escape_rtf_text(heading) + r"\b0\par")
            if remainder:
                rtf_content.append(escape_rtf_text(remainder) + r"\par")
            rtf_content.extend(build_summary_table(summary_data))
        
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

    @staticmethod
    def _speaker_sort_key(label: str) -> tuple[int, str]:
        match = re.match(r"^SPEAKER_(\d+)$", label)
        if match:
            return (0, f"{int(match.group(1)):03d}")
        return (1, label)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        if seconds <= 0.05:
            return "0s"
        if seconds >= 9.95:
            return f"{seconds:.0f}s"
        if seconds >= 1.0:
            return f"{seconds:.1f}s"
        return f"{seconds:.2f}s"

    @staticmethod
    def _format_with_percentage(value: float, total: float, *, decimals: int = 1, as_int: bool = False) -> str:
        if as_int:
            base_value = str(int(round(value)))
        else:
            base_value = f"{value:.{decimals}f}"

        if total <= 0 or value <= 0:
            return base_value

        percentage = (value / total) * 100.0
        return f"{base_value} ({percentage:.1f}%)"

    @classmethod
    def _get_speaker_summary_table_data(
        cls,
        segments: List[Dict[str, Any]],
        speaker_stats: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        metrics: Dict[str, Dict[str, float]] = {}

        def ensure_entry(label: Optional[str]) -> Dict[str, float]:
            normalized = label or 'UNKNOWN'
            entry = metrics.get(normalized)
            if entry is None:
                entry = {'words': 0.0, 'speech': 0.0, 'pauses': 0.0, 'turns': 0.0}
                metrics[normalized] = entry
            return entry

        for segment in segments:
            segment_speaker = segment.get('speaker') or 'UNKNOWN'
            if segment_speaker != 'SILENCE':
                ensure_entry(segment_speaker)['turns'] += 1.0

            words = segment.get('words', []) or []
            if not words:
                continue

            for word in words:
                if not isinstance(word, dict):
                    continue

                raw_speaker = word.get('speaker')
                speaker_label = raw_speaker or segment_speaker
                if not speaker_label or speaker_label == 'SILENCE':
                    speaker_label = segment_speaker if segment_speaker != 'SILENCE' else 'UNKNOWN'
                if not speaker_label or speaker_label == 'SILENCE':
                    continue

                entry = ensure_entry(speaker_label)

                start = word.get('start')
                end = word.get('end')
                duration = 0.0
                if start is not None and end is not None:
                    try:
                        duration = max(0.0, float(end) - float(start))
                    except (TypeError, ValueError):
                        duration = 0.0

                if _is_silence_token(word):
                    entry['pauses'] += duration
                    continue

                entry['words'] += 1
                entry['speech'] += duration

        if speaker_stats:
            for label in speaker_stats.keys():
                if label == 'SILENCE':
                    continue
                ensure_entry(label)

        non_silence_segments = [
            seg for seg in segments
            if not seg.get('is_silence_marker', False)
        ]
        non_silence_segments.sort(key=lambda seg: float(seg.get('start', 0.0) or 0.0))

        gap_total = 0.0
        for prev_seg, next_seg in zip(non_silence_segments, non_silence_segments[1:]):
            prev_speaker = prev_seg.get('speaker') or 'UNKNOWN'
            next_speaker = next_seg.get('speaker') or 'UNKNOWN'

            if prev_speaker in ('SILENCE', None) or next_speaker in ('SILENCE', None):
                continue
            if prev_speaker == next_speaker:
                continue

            prev_end = prev_seg.get('end')
            next_start = next_seg.get('start')
            try:
                prev_end_f = float(prev_end) if prev_end is not None else 0.0
                next_start_f = float(next_start) if next_start is not None else prev_end_f
            except (TypeError, ValueError):
                continue

            gap = next_start_f - prev_end_f
            if gap > 0:
                gap_total += gap

        if not metrics:
            return None

        speaker_names = [name for name in metrics.keys() if name != 'UNKNOWN']
        speaker_names.sort(key=cls._speaker_sort_key)
        if 'UNKNOWN' in metrics:
            speaker_names.append('UNKNOWN')

        rows: List[Dict[str, Any]] = []
        total_words = 0.0
        total_speech = 0.0
        total_pauses = 0.0
        total_turns = 0.0

        for name in speaker_names:
            data = metrics[name]
            words = int(round(data.get('words', 0.0)))
            speech = float(data.get('speech', 0.0))
            pauses = float(data.get('pauses', 0.0))
            turns = float(data.get('turns', 0.0))
            speed = (words / speech * 60.0) if speech > 0 else 0.0
            words_per_turn_value = (words / turns) if turns > 0 else 0.0

            rows.append(
                {
                    'speaker': name,
                    'words_value': float(words),
                    'speech_value': speech,
                    'turns_value': turns,
                    'words_per_turn_value': words_per_turn_value,
                    'speed': f"{int(round(speed))} wpm" if speech > 0 else "--",
                    'pauses': cls._format_seconds(pauses),
                }
            )

            total_words += words
            total_speech += speech
            total_pauses += pauses
            total_turns += turns

        gaps_row = None
        if gap_total > 0:
            gaps_row = {
                'speaker': 'GAPS',
                'turns': '',
                'words': '',
                'words_per_turn': '',
                'speech': '',
                'speed': '',
                'pauses': cls._format_seconds(gap_total),
            }
            total_pauses += gap_total

        total_row = None
        if rows or gap_total > 0:
            total_speed = (total_words / total_speech * 60.0) if total_speech > 0 else 0.0
            total_duration = total_speech + total_pauses
            silence_pct = (total_pauses / total_duration * 100.0) if total_duration > 0 else 0.0
            if total_pauses > 0:
                pauses_text = f"{cls._format_seconds(total_pauses)} ({silence_pct:.1f}%)"
            else:
                pauses_text = cls._format_seconds(total_pauses)
            words_per_turn_total = (total_words / total_turns) if total_turns > 0 else 0.0
            total_row = {
                'speaker': 'Total',
                'turns_value': float(total_turns),
                'words_value': float(total_words),
                'speech_value': total_speech,
                'words_per_turn_value': words_per_turn_total,
                'speed': f"{int(round(total_speed))} wpm" if total_speech > 0 else "--",
                'pauses': pauses_text,
            }

        def format_words(value: float) -> str:
            return cls._format_with_percentage(value, total_words, as_int=True)

        def format_speech(value: float) -> str:
            return cls._format_with_percentage(value, total_speech, decimals=1)

        def format_turns(value: float) -> str:
            return cls._format_with_percentage(value, total_turns, as_int=True)

        display_rows: List[Dict[str, str]] = []
        for row in rows:
            display_rows.append(
                {
                    'speaker': row['speaker'],
                    'turns': format_turns(row['turns_value']),
                    'words': format_words(row['words_value']),
                    'words_per_turn': f"{row['words_per_turn_value']:.1f}" if row['words_per_turn_value'] > 0 else '--',
                    'speech': format_speech(row['speech_value']),
                    'speed': row['speed'],
                    'pauses': row['pauses'],
                }
            )

        display_gaps = gaps_row

        display_total = None
        if total_row:
            display_total = {
                'speaker': total_row['speaker'],
                'turns': format_turns(total_row['turns_value']),
                'words': format_words(total_row['words_value']),
                'words_per_turn': f"{total_row['words_per_turn_value']:.1f}" if total_row['words_per_turn_value'] > 0 else '--',
                'speech': format_speech(total_row['speech_value']),
                'speed': total_row['speed'],
                'pauses': total_row['pauses'],
            }

        columns = [
            ('speaker', '', 'left'),
            ('turns', 'TURNS', 'right'),
            ('words', 'WORDS', 'right'),
            ('words_per_turn', 'WORD/TURN', 'right'),
            ('speech', 'SPEECH', 'right'),
            ('speed', 'SPEED', 'right'),
            ('pauses', 'PAUSES', 'right'),
        ]

        width_rows: List[Dict[str, str]] = list(display_rows)
        if display_gaps:
            width_rows.append(display_gaps)
        if display_total:
            width_rows.append(display_total)

        if not width_rows:
            return None

        column_widths: Dict[str, int] = {}
        for key, label, _alignment in columns:
            max_length = len(label)
            for row in width_rows:
                cell_value = row.get(key, '') if row else ''
                if cell_value is None:
                    cell_value = ''
                max_length = max(max_length, len(cell_value))
            column_widths[key] = max_length

        return {
            'columns': columns,
            'rows': display_rows,
            'gap_row': display_gaps,
            'total_row': display_total,
            'column_widths': column_widths,
        }

    @classmethod
    def _build_speaker_summary(
        cls,
        segments: List[Dict[str, Any]],
        speaker_stats: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[str]:
        table_data = cls._get_speaker_summary_table_data(segments, speaker_stats)
        if not table_data:
            return []

        columns = table_data['columns']
        display_rows = table_data['rows']
        display_gaps = table_data['gap_row']
        display_total = table_data['total_row']
        column_widths = table_data['column_widths']

        def format_line(row: Dict[str, str]) -> str:
            parts: List[str] = []
            for key, _label, alignment in columns:
                width = column_widths[key]
                value = row.get(key, '') or ''
                if alignment == 'left':
                    parts.append(f"{value:<{width}}")
                elif alignment == 'center':
                    parts.append(f"{value:^{width}}")
                else:
                    parts.append(f"{value:>{width}}")
            return "  ".join(parts)

        header_parts: List[str] = []
        for key, label, alignment in columns:
            width = column_widths[key]
            if alignment == 'left':
                header_parts.append(f"{label:<{width}}")
            elif alignment == 'center':
                header_parts.append(f"{label:^{width}}")
            else:
                header_parts.append(f"{label:>{width}}")
        header = "  ".join(header_parts)

        lines = [SPEAKER_STATS_WARNING, "", header, ""]
        for row in display_rows:
            lines.append(format_line(row))

        if display_gaps:
            lines.append(format_line(display_gaps))

        if display_total:
            lines.append(format_line(display_total))

        return lines

    def format(self, segments: List[Dict], output_path: str, include_silence: bool = True, config=None, transcript_id: str = None, speaker_stats: Optional[Dict[str, Dict[str, Any]]] = None, **kwargs) -> None:
        """Create a plain text transcript with speaker paragraphs."""
        gap_threshold = getattr(config, 'silence_gap_linebreak_threshold', None) if config else None

        paragraphs: List[Dict[str, Any]] = []
        current_speaker = None
        current_paragraph: List[str] = []
        current_start_time = None

        def smart_join(parts: List[str]) -> str:
            if not parts:
                return ""
            result = parts[0]
            for i in range(1, len(parts)):
                prev_part = parts[i - 1]
                curr_part = parts[i]
                
                # Check if we need a space separator
                # Add space unless the previous part ends with whitespace (including newlines)
                # or the current part starts with whitespace
                needs_space = True
                if prev_part and prev_part[-1] in (' ', '\n', '\t', '\r'):
                    needs_space = False
                if curr_part and curr_part[0] in (' ', '\n', '\t', '\r'):
                    needs_space = False
                
                if needs_space:
                    result += ' ' + curr_part
                else:
                    result += curr_part
            return result

        for segment in segments:
            is_silence = segment.get('is_silence_marker', False)
            speaker = segment.get('speaker', 'UNKNOWN')

            text = WordProcessor.create_paragraph_text_from_words(segment, gap_threshold=gap_threshold)
            if not text:
                continue

            if is_silence:
                if current_paragraph:
                    # For silence segments, find the first silence marker word to get duration
                    duration = None
                    words = segment.get('words', [])
                    for word in words:
                        if word.get('is_silence_marker', False):
                            duration = WordProcessor._parse_silence_duration(word.get('word', ''))
                            break
                    
                    if duration is not None and gap_threshold and duration >= gap_threshold:
                        formatted_text = f"\n\n{text}\n\n"
                    else:
                        formatted_text = text
                    current_paragraph.append(formatted_text)
                continue

            if speaker != current_speaker:
                if current_paragraph and current_speaker is not None:
                    paragraph_text = smart_join(current_paragraph)
                    paragraphs.append(
                        {
                            'speaker': current_speaker,
                            'text': paragraph_text,
                            'start_time': current_start_time,
                        }
                    )

                current_speaker = speaker
                current_paragraph = [text]
                current_start_time = segment.get('start', 0)
            else:
                current_paragraph.append(text)

        if current_paragraph and current_speaker is not None:
            paragraph_text = smart_join(current_paragraph)
            paragraphs.append(
                {
                    'speaker': current_speaker,
                    'text': paragraph_text,
                    'start_time': current_start_time,
                }
            )

        txt_content: List[str] = []

        summary_lines = self._build_speaker_summary(segments, speaker_stats)

        if config and hasattr(config, 'output_preamble') and config.output_preamble:
            if transcript_id and hasattr(config, 'get_preamble_with_transcript_id'):
                preamble_text = config.get_preamble_with_transcript_id(transcript_id)
            else:
                preamble_text = config.output_preamble
            txt_content.append(preamble_text)
            txt_content.append("")

            if summary_lines:
                txt_content.extend(summary_lines)
                txt_content.append("")

            txt_content.append("=" * 90) # separator line
            txt_content.append("")
        elif summary_lines:
            txt_content.extend(summary_lines)
            txt_content.append("")

        for paragraph in paragraphs:
            speaker = paragraph['speaker']
            text = paragraph['text']
            start_time = paragraph['start_time'] or 0

            hours = int(start_time // 3600)
            minutes = int((start_time % 3600) // 60)
            seconds = int(start_time % 60)
            timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"

            txt_content.append(f"{speaker} {timestamp}:")
            txt_content.append(text)
            txt_content.append("")

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
            segments: List of segment dictionaries with words and metadata
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
