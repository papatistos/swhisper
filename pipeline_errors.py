"""Short, actionable failure messages for pipeline output."""
from pathlib import Path


def failure_reason(error):
    if isinstance(error, FileNotFoundError) and error.filename:
        path = Path(error.filename)
        resolved = path.resolve()
        if resolved != path and not resolved.exists():
            return (f"Model or file path is unavailable: {path} points to {resolved}. "
                    "Reconnect the drive if needed, or repair the cache symlink.")
    return f"{type(error).__name__}: {error}"


def last_failure(lines):
    """Use the application's explicit error, not cleanup or history messages."""
    for line in reversed(lines):
        if line.startswith('Error: '):
            return line[len('Error: '):].strip()
    return None
