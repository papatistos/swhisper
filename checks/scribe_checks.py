"""Offline contract checks: python -m unittest discover -s checks -p '*_checks.py'."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from types import SimpleNamespace

from scribe_api import ScribeClient, ScribeError, normalize_result
import transcribe_scribe


TRANSCRIPT = {"full_transcription": "Hej, ja!", "segments": [
    {"start": 0, "end": 1, "text": "Hej,", "speaker": "Speaker_00"},
    {"start": 1, "end": 2, "text": "ja!", "speaker": "Speaker_01"},
]}
WORDS = {"version": 1, "words": [
    {"t": "Hej,", "s": 0.1, "e": 0.6, "c": 0.9},
    {"t": "ja!", "s": 1.1, "e": 1.4},
]}


class Response:
    def __init__(self, result, status=200):
        self.status_code = status
        self.result = result

    def json(self):
        return {"result": self.result}


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        if "data" in kwargs:
            body = kwargs["data"]
            actual = b"".join(body)
            assert len(actual) == len(body)
            assert b'name="file"' in actual
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class AdapterChecks(unittest.TestCase):
    def test_preserves_words_and_timings_without_remote_speakers(self):
        result = normalize_result(TRANSCRIPT, WORDS)
        words = [w for s in result["segments"] for w in s["words"]]
        self.assertEqual([w["text"] for w in words], ["Hej,", "ja!"])
        self.assertEqual([(w["start"], w["end"]) for w in words], [(0.1, 0.6), (1.1, 1.4)])
        self.assertNotIn("confidence", words[1])
        self.assertNotIn("speaker", result["segments"][0])
        self.assertNotIn("speaker", words[0])

    def test_fails_on_missing_unknown_mismatched_or_invalid_words(self):
        variants = [dict(version=2, words=WORDS["words"]), dict(version=1, words=[])]
        for field, value in [("t", "Missing"), ("s", float("nan")), ("e", -1), ("c", 2)]:
            data = copy.deepcopy(WORDS)
            data["words"][0][field] = value
            variants.append(data)
        for data in variants:
            with self.subTest(data=data), self.assertRaises(ScribeError):
                normalize_result(TRANSCRIPT, data)

    def test_boundary_straddling_word_is_not_lost_or_duplicated(self):
        words = copy.deepcopy(WORDS)
        words["words"][0].update(s=0.9, e=1.1)
        result = normalize_result(TRANSCRIPT, words)
        flattened = [w["text"] for s in result["segments"] for w in s["words"]]
        self.assertEqual(flattened, ["Hej,", "ja!"])


class ClientChecks(unittest.TestCase):
    def test_resume_completed_job_never_uploads_or_queues(self):
        session = Session([Response({"status": "completed"}),
                           Response(json.dumps(TRANSCRIPT)), Response(json.dumps(WORDS))])
        client = ScribeClient("secret-token", "secret-password", session=session)
        transcript, words = client.transcribe(None, {"job_id": "abc"}, None, "Swedish (verbatim)")
        self.assertEqual(normalize_result(transcript, words)["text"], "Hej, ja!")
        self.assertTrue(all(c[0] == "GET" for c in session.calls))
        self.assertEqual(session.calls[-1][2]["json"], {"encryption_password": "secret-password"})

    def test_upload_queue_and_persist_without_credentials(self):
        session = Session([Response({"uuid": "abc"}), Response({"status": "uploaded"}),
                           Response({"status": "pending"}), Response({"status": "completed"}),
                           Response(json.dumps(TRANSCRIPT)), Response(json.dumps(WORDS))])
        client = ScribeClient("secret-token", "secret-password", session=session)
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "audio.wav"
            wav.write_bytes(b"audio")
            state_path = Path(directory) / "state.json"
            with patch("scribe_api.time.sleep"):
                client.transcribe(wav, {}, state_path, "Swedish (verbatim)")
            state = state_path.read_text()
            self.assertIn("abc", state)
            self.assertNotIn("secret", state)
            self.assertEqual(session.calls[2][2]["json"]["language"], "Swedish (verbatim)")

    def test_ambiguous_upload_is_not_repeated(self):
        session = Session([])
        client = ScribeClient("token", "password", session=session)
        with self.assertRaisesRegex(ScribeError, "unknown outcome"):
            client.transcribe(None, {"upload_started": True}, None, "Swedish")
        self.assertEqual(session.calls, [])

    def test_auth_failure_preserves_job_and_hides_secrets(self):
        client = ScribeClient("secret-token", "secret-password", session=Session([Response({}, 401)]))
        state = {"job_id": "abc"}
        with self.assertRaises(ScribeError) as error:
            client.transcribe(None, state, None, "Swedish")
        self.assertNotIn("secret", str(error.exception))
        self.assertEqual(state, {"job_id": "abc"})

    def test_auth_rejection_clears_definitive_upload_marker(self):
        client = ScribeClient("token", "password", session=Session([Response({}, 401)]))
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "audio.wav"
            wav.write_bytes(b"audio")
            state = {}
            with self.assertRaises(ScribeError):
                client.transcribe(wav, state, Path(directory) / "state.json", "Swedish")
            self.assertNotIn("upload_started", state)

    def test_encrypted_payload_is_rejected(self):
        client = ScribeClient("token", "password", session=Session([Response("ciphertext")]))
        with self.assertRaisesRegex(ScribeError, "encrypted"):
            client.result("abc", "/words")


class PrototypeChecks(unittest.TestCase):
    def test_local_handoff_forces_scope_after_sidecar(self):
        import diarize4
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "interview.wav"
            source.with_suffix(".swhisper").write_text(
                "SWHISPER_BACKFILL_ENABLED=true\nSWHISPER_NUM_SPEAKERS=2\n"
            )
            work = Path(directory) / "run"
            processor = Mock(return_value=True)
            module = SimpleNamespace(process_file=processor)
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
            with patch.dict("os.environ", {"TRANSCRIBE_AUDIO_DIR": "/wrong", "JSON_INPUT_DIR": "/wrong", "AUDIO_SUBDIR": "/wrong"}), \
                    patch.object(diarize4, "ensure_pyannote_v4"), \
                    patch("transcribe_scribe.importlib.util.spec_from_file_location", return_value=spec), \
                    patch("transcribe_scribe.importlib.util.module_from_spec", return_value=module):
                transcribe_scribe.local_diarization(source, work, "audio.json")
            config = processor.call_args.args[0]
            self.assertFalse(config.backfill_missing_turns)
            self.assertFalse(config.backfill_enable_sensevoice)
            self.assertFalse(config.use_precision_service)
            self.assertEqual(config.whisper_timestamp_offset, 0)
            self.assertEqual(config.min_speakers, 2)
            self.assertEqual(config.max_speakers, 2)
            self.assertEqual(config.final_audio_dir, str(work))
            self.assertEqual(config.final_json_input_dir, str(work / "whisper-json-output"))
            self.assertEqual(config.final_output_dir, str(work / "transcripts"))

    def test_transcription_only_and_cached_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "original.wav"
            source.write_bytes(b"original audio")
            work = Path(directory) / "output"
            def prepare(source, target):
                target.write_bytes(b"prepared audio")
            def transcribe(client, wav, state, state_path, language):
                state["job_id"] = "abc"
                return TRANSCRIPT, WORDS
            argv = [str(source), "--work-dir", str(work), "--transcription-only", "--auth", "env"]
            with patch.dict("os.environ", {"SUNET_SCRIBE_API_KEY": "api-key", "SWHISPER_SCRIBE_TOKEN": "token", "SWHISPER_SCRIBE_PASSWORD": "password"}), \
                    patch.object(transcribe_scribe, "prepare_audio", side_effect=prepare), \
                    patch.object(ScribeClient, "transcribe", autospec=True, side_effect=transcribe) as remote, \
                    patch.object(transcribe_scribe, "local_diarization") as local:
                self.assertEqual(transcribe_scribe.main(argv), 0)
                self.assertEqual(transcribe_scribe.main(argv), 0)
                self.assertEqual(remote.call_count, 1)
                self.assertEqual(remote.call_args.args[0].token, "api-key")
                local.assert_not_called()
                self.assertEqual(source.read_bytes(), b"original audio")
                source.write_bytes(b"changed")
                self.assertEqual(transcribe_scribe.main(argv), 1)
                self.assertEqual(remote.call_count, 1)


if __name__ == "__main__":
    unittest.main()
