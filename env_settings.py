"""Read and update selected env-file keys without discarding unrelated text."""
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def parse_env(text):
    values = {}
    for line in text.splitlines():
        match = re.match(r'^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=(.*)$', line)
        if not match:
            continue
        key, raw = match[1], match[2].strip()
        try:
            if raw.startswith('"'):
                value, _ = json.JSONDecoder().raw_decode(raw)
                if not isinstance(value, str):
                    continue
            elif raw.startswith("'") and "'" in raw[1:]:
                value = raw[1:raw.rfind("'")]
                value = value.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
            else:
                value = raw.split('#', 1)[0].strip()
                value = value.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
        except ValueError:
            value = raw.strip('"').strip("'")
            value = value.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
        values[key] = value
    return values


def env_snapshot(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8') if path.exists() else ''
    return text, hashlib.sha256(text.encode()).hexdigest()


def update_env(path, values, managed_keys, revision):
    """Replace managed assignments atomically; all other lines remain verbatim."""
    path = Path(path)
    text, current = env_snapshot(path)
    if current != revision:
        raise ValueError('Global settings changed since you opened them. Reload before saving.')
    lines, seen = [], set()
    for line in text.splitlines(keepends=True):
        match = re.match(r'^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=', line)
        key = match[1] if match else None
        if key not in managed_keys:
            lines.append(line)
            continue
        if key in seen:
            continue
        seen.add(key)
        # Preserve unchanged assignments, including their original comments.
        if key in values and parse_env(line).get(key) == values[key]:
            lines.append(line)
        elif key in values:
            raw = line.split('=', 1)[1].strip()
            if raw.startswith('"'):
                try:
                    _, end = json.JSONDecoder().raw_decode(raw)
                    comment = raw[end:].strip()
                except ValueError:
                    comment = ''
            elif raw.startswith("'"):
                comment = raw[raw.rfind("'") + 1:].strip()
            else:
                comment = '#' + raw.split('#', 1)[1] if '#' in raw else ''
            lines.append(f'{key}={json.dumps(values[key], ensure_ascii=False)}' + (f'  {comment}' if comment.startswith('#') else '') + '\n')
        else:
            lines.append('# Using program default: ' + line.rstrip('\n') + '\n')
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n'
    for key in sorted(set(values) - seen):
        lines.append(f'{key}={json.dumps(values[key], ensure_ascii=False)}\n')
    fd, temporary = tempfile.mkstemp(prefix='.swhisper-settings-', dir=path.parent)
    try:
        if path.exists():
            os.fchmod(fd, path.stat().st_mode & 0o777)
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(''.join(lines))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
