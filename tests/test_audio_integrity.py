import hashlib
import json
import struct
import wave
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from audio_integrity import pcm_checksum, checksum_without_jpr_metadata
import transcribe_scribe


@pytest.mark.parametrize('legacy', [True, False])
@pytest.mark.parametrize('samples_changed', [True, False])
def test_app_metadata_does_not_trigger_rebuild_but_sample_changes_do(tmp_path, legacy, samples_changed):
    source = tmp_path / 'source.m4a'
    source.write_bytes(b'original recording')
    audio = tmp_path / 'audio.wav'
    with wave.open(str(audio), 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        wav.writeframes(b'\x01\x00' * 100)
    original = audio.read_bytes()
    expected = hashlib.sha256(original).hexdigest()
    pcm = pcm_checksum(audio)
    marker = b'{"_class":"JPRAudioFileUserData","playcnt":0}'
    appended = b'JPR2' + struct.pack('<I', len(marker)) + marker + b'\x00' * (len(marker) % 2)
    altered = bytearray(original + appended)
    altered[4:8] = struct.pack('<I', len(altered) - 8)
    if samples_changed:
        altered[44] ^= 1
    audio.write_bytes(altered)
    assert (pcm_checksum(audio) == pcm) == (not samples_changed)
    assert (checksum_without_jpr_metadata(audio) == expected) == (not samples_changed)
    transcript = tmp_path / 'whisper-json-output/audio.json'
    transcript.parent.mkdir()
    transcript.write_text('saved')
    args = SimpleNamespace(language='Swedish', base_url='https://scribe.sunet.se', job_id=None, transcription_only=True)
    state = {'identity': {'source': str(source), 'sha256': transcribe_scribe.sha256(source),
                         'language': args.language, 'base_url': args.base_url, 'adapter_version': 1,
                         'audio_format': 'pcm_s16le/16000/mono'},
             'wav_sha256': expected, 'output_sha256': transcribe_scribe.sha256(transcript)}
    if not legacy:
        state['local_pcm_sha256'] = pcm
    (tmp_path / 'scribe-state.json').write_text(json.dumps(state))
    with patch.object(transcribe_scribe, 'restore_working_audio') as rebuild, \
            patch.object(transcribe_scribe, 'ScribeClient') as remote:
        transcribe_scribe.run_locked(args, source, tmp_path)
    assert rebuild.call_count == int(samples_changed)
    remote.assert_not_called()
    assert audio.read_bytes() == altered
    assert transcript.read_text() == 'saved'
