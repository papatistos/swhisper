"""Time anchors read back out of a stored TXT transcript.

A transcript with a diarized JSON beside it gets exact anchors from
``TXTFormatter.render_with_anchors``. Older transcripts have only the TXT, but
every paragraph header still carries its start time, which is precise enough
to line two transcripts up by time.
"""

import re
from typing import Any, Dict, List

# "SPEAKER_00 [00:01:02]:" or a renamed speaker such as "Anna Berg [00:01:02]:"
_HEADER = re.compile(r'^\S.* \[(\d{2,}):(\d{2}):(\d{2})\]:$')


def anchors_from_txt(content: str) -> List[Dict[str, Any]]:
    """Return the paragraph anchors of a TXT transcript, in the formatter's shape.

    A header carries only its start, so each paragraph is taken to end where
    the next one begins.
    """
    anchors: List[Dict[str, Any]] = []
    for line, text in enumerate(content.split('\n')):
        match = _HEADER.match(text)
        if match:
            hours, minutes, seconds = (int(part) for part in match.groups())
            start = float(hours * 3600 + minutes * 60 + seconds)
            if anchors:
                anchors[-1]['end'] = max(anchors[-1]['start'], start)
            anchors.append({'line': line, 'start': start, 'end': start})
    return anchors
