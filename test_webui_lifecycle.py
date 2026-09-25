"""Restart commands must preserve jobs and refuse unrelated listeners."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.request import ProxyHandler, build_opener

import webui_lifecycle as lifecycle


SCRIPT = Path(__file__).resolve().with_name("webui.py")


class ProcessSafetyTests(unittest.TestCase):
    def discover(self, command, cwd=None, uid=None):
        identity = f"{os.getuid() if uid is None else uid} Fri Sep 25 02:41:40 2026 {command}"
        with patch.object(lifecycle, "_command", side_effect=[
            "p123", identity, f"p123\nfcwd\nn{cwd or SCRIPT.parent}",
        ]):
            return lifecycle.find_instance("127.0.0.1", 8722, SCRIPT)

    def test_recognizes_existing_server_without_pid_file(self):
        self.assertEqual(self.discover("/env/bin/python -u webui.py --no-browser").pid, 123)
        self.assertEqual(self.discover(f"/env/bin/python {SCRIPT}").pid, 123)

    def test_rejects_other_application_checkout_and_user(self):
        cases = [
            ("python -m http.server 8722", None, None),
            ('python -c "print(\'webui.py\')"', None, None),
            ("python webui.py", "/tmp/other-checkout", None),
            ("python webui.py", None, os.getuid() + 1),
        ]
        for command, cwd, uid in cases:
            with self.subTest(command=command, cwd=cwd, uid=uid):
                with self.assertRaises(lifecycle.LifecycleError):
                    self.discover(command, cwd, uid)

    def test_changed_listener_is_never_signalled(self):
        old = lifecycle.Instance(123, "old start time")
        with patch.object(lifecycle, "find_instance", return_value=lifecycle.Instance(123, "new start time")), \
                patch.object(lifecycle.os, "kill") as kill:
            with self.assertRaises(lifecycle.LifecycleError):
                lifecycle.stop_instance(old, "127.0.0.1", 8722, SCRIPT)
            kill.assert_not_called()

    def test_stop_timeout_does_not_force_kill(self):
        instance = lifecycle.Instance(123, "identity")
        with patch.object(lifecycle, "find_instance", return_value=instance), \
                patch.object(lifecycle.os, "kill") as kill:
            with self.assertRaisesRegex(lifecycle.LifecycleError, "not force-killed"):
                lifecycle.stop_instance(instance, "127.0.0.1", 8722, SCRIPT, timeout=0)
            kill.assert_called_once_with(123, lifecycle.signal.SIGTERM)


@unittest.skipUnless(shutil.which("lsof") and shutil.which("ps"), "requires local process tools")
class LiveLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.audio = Path(self.temp.name)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.command = [sys.executable, str(SCRIPT), "--port", str(self.port),
                        "--audio-dir", str(self.audio), "--no-browser"]
        self.http = build_opener(ProxyHandler({}))

    def process(self, command):
        process = subprocess.Popen(command, cwd=SCRIPT.parent, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        def cleanup():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self.addCleanup(cleanup)
        return process

    def cli(self, *args):
        return subprocess.run(self.command + list(args), capture_output=True, text=True, timeout=20)

    def ready(self, process):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.assertIsNone(process.poll(), "WebUI exited before serving")
            try:
                with self.http.open(f"http://127.0.0.1:{self.port}/api/run/status", timeout=.5) as response:
                    return json.load(response)
            except OSError:
                time.sleep(.1)
        self.fail("WebUI did not become ready")

    def test_start_duplicate_restart_stop_preserves_and_reattaches_job(self):
        job = self.process([sys.executable, "-c", "import time; time.sleep(90)"])
        logs = self.audio / "logs"
        logs.mkdir()
        (logs / "ui-run-test.log").write_text("Test job is still running\n")
        (logs / "ui-run-active.json").write_text(json.dumps({
            "pid": job.pid, "run_id": "restart-test", "pipeline": "diarize4",
            "audio_dir": str(self.audio), "started_at": time.time(),
            "log_path": "logs/ui-run-test.log",
        }))
        first = self.process(self.command)
        self.assertTrue(self.ready(first)["running"])
        duplicate = self.cli()
        self.assertEqual(duplicate.returncode, 0, duplicate.stderr)
        self.assertIn("already running", duplicate.stdout)
        self.assertIn("--restart", duplicate.stdout)
        self.assertNotIn("on_event", duplicate.stderr)
        self.assertIn(str(first.pid), self.cli("--status").stdout)
        replacement = self.process(self.command + ["--restart"])
        first.wait(timeout=15)
        status = self.ready(replacement)
        self.assertTrue(status["running"])
        self.assertEqual(status["run_id"], "restart-test")
        self.assertIsNone(job.poll())
        self.assertIn(str(replacement.pid), self.cli("--status").stdout)
        stopped = self.cli("--stop")
        self.assertEqual(stopped.returncode, 0, stopped.stderr)
        replacement.wait(timeout=5)
        self.assertIsNone(job.poll())
        self.assertIn("not running", self.cli("--status").stdout)
        self.assertEqual(self.cli("--stop").returncode, 0)

    def test_unrelated_listener_is_not_stopped(self):
        other = self.process([sys.executable, "-m", "http.server", str(self.port),
                              "--bind", "127.0.0.1", "--directory", str(self.audio)])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=.2):
                    break
            except OSError:
                time.sleep(.1)
        for action in ("--restart", "--stop"):
            result = self.cli(action)
            self.assertEqual(result.returncode, 1)
            self.assertIn("another application", result.stderr)
            self.assertIsNone(other.poll())
