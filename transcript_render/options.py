"""The knobs a reader can turn without re-running the pipeline."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RenderOptions:
    """How to present an already-diarized transcript.

    Every field here is recoverable from the diarized JSON, so switching any of
    them is a re-render rather than a re-run. The two settings that are *not*
    here — ``SWHISPER_DETECT_DISFLUENCIES`` and ``SWHISPER_PRESERVE_MARKERS`` —
    change what data exists in the first place and stay pipeline settings.
    """

    # Show the [*] / [S02**] markers for sounds that could not be transcribed.
    # Hiding them merges each marker's span into the pauses on either side,
    # exactly as running the pipeline with preserve_markers off would.
    disfluencies: bool = True

    # Show the (1.3) pause durations.
    pauses: bool = True

    # Put blank lines around pauses at or above pause_linebreak_threshold
    # seconds. Independent of `pauses`: with pauses hidden this still breaks
    # the paragraph, it just does not print the number.
    pause_linebreaks: bool = True
    pause_linebreak_threshold: float = 1.0

    # Hide pauses shorter than this. 0 shows every pause the run recorded;
    # the run's own threshold is the floor, so this can only raise the bar.
    min_pause: float = 0.0

    # Header blocks.
    preamble: bool = True
    stats: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RenderOptions":
        """Build from a JSON body, ignoring unknown keys and bad values."""
        if not data:
            return cls()
        fields = cls.__dataclass_fields__
        kwargs: Dict[str, Any] = {}
        for key, field in fields.items():
            if key not in data:
                continue
            raw = data[key]
            if field.type == 'bool' or isinstance(field.default, bool):
                if isinstance(raw, str):
                    kwargs[key] = raw.strip().lower() in ('1', 'true', 'yes', 'on')
                else:
                    kwargs[key] = bool(raw)
            else:
                try:
                    kwargs[key] = max(0.0, float(raw))
                except (TypeError, ValueError):
                    continue
        return cls(**kwargs)

    @property
    def linebreak_threshold(self) -> Optional[float]:
        """The threshold to hand the paragraph builder, or None to disable."""
        if not self.pause_linebreaks or self.pause_linebreak_threshold <= 0:
            return None
        return self.pause_linebreak_threshold

    def suffix(self) -> str:
        """A short filename tag describing this view, empty for the default."""
        parts = []
        if not self.disfluencies:
            parts.append("nodisfl")
        if not self.pauses:
            parts.append("nopause")
        if not self.pause_linebreaks:
            parts.append("nobreaks")
        elif self.pause_linebreak_threshold != 1.0:
            parts.append(f"brk{self.pause_linebreak_threshold:g}".replace('.', ''))
        if self.min_pause > 0:
            parts.append(f"min{self.min_pause:g}".replace('.', ''))
        if not self.stats:
            parts.append("nostats")
        return ("_" + "_".join(parts)) if parts else ""
