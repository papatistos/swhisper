"""Verify PCM audio independently of mutable WAV application metadata."""
import hashlib
import json
import struct
import wave


def pcm_checksum(path):
    try:
        with wave.open(str(path), 'rb') as audio:
            digest = hashlib.sha256(json.dumps([
                audio.getnchannels(), audio.getsampwidth(), audio.getframerate(),
                audio.getnframes(), audio.getcomptype(),
            ]).encode())
            for block in iter(lambda: audio.readframes(65536), b''):
                digest.update(block)
            return digest.hexdigest()
    except (wave.Error, EOFError):
        return None


def checksum_without_jpr_metadata(path):
    """Recover a legacy file checksum virtually; never alter the WAV on disk."""
    with open(path, 'rb') as audio:
        header = audio.read(12)
        if len(header) != 12 or header[:4] != b'RIFF' or header[8:] != b'WAVE':
            return None
        audio.seek(0, 2)
        size = audio.tell()
        position, spans, removed = 12, [], False
        while position < size:
            audio.seek(position)
            chunk = audio.read(8)
            if len(chunk) != 8:
                return None
            tag, length = struct.unpack('<4sI', chunk)
            length = 8 + length + length % 2
            if position + length > size:
                return None
            if tag == b'JPR2':
                removed = True
            else:
                spans.append((position, length))
            position += length
        if not removed:
            return None
        digest = hashlib.sha256(b'RIFF' + struct.pack('<I', 4 + sum(n for _, n in spans)) + b'WAVE')
        for offset, length in spans:
            audio.seek(offset)
            while length:
                block = audio.read(min(length, 1048576))
                if not block:
                    return None
                digest.update(block)
                length -= len(block)
        return digest.hexdigest()
