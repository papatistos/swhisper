#!/usr/bin/env python3
"""
swhisper - Speaker diarization entry point for pyannote.audio 4 pipelines.

This script keeps the legacy diarization workflow but switches the underlying
pyannote pipeline to the community model that requires pyannote.audio >= 4.
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure local imports resolve identically to the legacy script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diarize import DiarizationConfig


def _invoke_legacy_main(config: DiarizationConfig) -> None:
    """Load diarize.py dynamically so we can pass configuration objects."""
    module_path = Path(__file__).with_name("diarize.py")
    spec = importlib.util.spec_from_file_location("swhisper_diarize_legacy", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load diarize.py entry point.")

    legacy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_module)

    legacy_main = getattr(legacy_module, "main", None)
    if legacy_main is None:
        raise AttributeError("diarize.py does not define a main() function.")

    legacy_main(config)

LEGACY_PIPELINE_ID = "pyannote/speaker-diarization-3.1"
COMMUNITY_PIPELINE_ID = "pyannote/speaker-diarization-community-1"
PRECISION_PIPELINE_ID = "pyannote/speaker-diarization-precision-2"
PRECISION_TOKEN_ENV_VARS = ("PYANNOTEAI_API_KEY", "PYANNOTE_API_KEY")


def resolve_pyannote_token(existing_token: str = "") -> str:
    """Pick the best available Hugging Face token."""
    for env_var in ("HUGGINGFACE_ACCESS_TOKEN", "HUGGING_FACE_TOKEN", "PYANNOTE_TOKEN"):
        token = os.getenv(env_var, "").strip()
        if token:
            return token
    return existing_token.strip()


def resolve_precision_token(existing_token: str = "") -> str:
    """Resolve pyannote precision API token from known environment variables."""
    for env_var in PRECISION_TOKEN_ENV_VARS:
        token = os.getenv(env_var, "").strip()
        if token:
            return token
    return existing_token.strip()


def ensure_pyannote_v4() -> None:
    """Verify that pyannote.audio 4.x is available."""
    try:
        import pyannote.audio  # type: ignore
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("pyannote.audio >= 4.0.0 is required for diarize4.py") from exc

    version_str = getattr(pyannote.audio, "__version__", "0.0")
    major_token = version_str.split(".")[0]

    try:
        major = int(major_token)
    except ValueError:  # pragma: no cover - unexpected version format
        major = 0

    if major < 4:
        raise RuntimeError(f"pyannote.audio >= 4.0.0 is required, found {version_str}")


def run(config: Optional[DiarizationConfig] = None, *, show_header: bool = True) -> None:
    """Execute diarization using the community pipeline."""
    if show_header:
        print("\n" + "=" * 60)
        print("🎯 Launching diarization pipeline (pyannote.audio 4)...")

    try:
        ensure_pyannote_v4()
    except RuntimeError as version_error:
        print(f"❌ {version_error}")
        return

    try:
        config_obj = config or DiarizationConfig()
    except Exception as exc:  # pragma: no cover - config validation guard
        print(f"❌ Unable to initialize diarization config: {exc}")
        return

    precision_pipeline_id = getattr(config_obj, "precision_pipeline_model", PRECISION_PIPELINE_ID)
    use_precision = getattr(config_obj, "use_precision_service", False)

    if use_precision and config_obj.pipeline_model != precision_pipeline_id:
        config_obj.pipeline_model = precision_pipeline_id

    is_precision = config_obj.pipeline_model == precision_pipeline_id

    if is_precision:
        config_obj.precision_api_token = resolve_precision_token(
            getattr(config_obj, "precision_api_token", "")
        )
        if not config_obj.precision_api_token:
            print(
                "⚠️ Precision diarization token not detected. Set PYANNOTEAI_API_KEY "
                "(or override precision_api_token) to use the precision-2 service."
            )
    else:
        if getattr(config_obj, "pipeline_model", "").strip() in {"", LEGACY_PIPELINE_ID}:
            config_obj.pipeline_model = COMMUNITY_PIPELINE_ID
        config_obj.hugging_face_token = resolve_pyannote_token(config_obj.hugging_face_token)
        if not config_obj.hugging_face_token:
            print(
                "⚠️ Hugging Face token not detected. Set HUGGINGFACE_ACCESS_TOKEN or "
                "HUGGING_FACE_TOKEN to access the community pipeline."
            )

    try:
        _invoke_legacy_main(config_obj)
    except SystemExit as exit_info:
        if exit_info.code not in (None, 0):
            print(f"❌ Diarization exited with status {exit_info.code}")
    except Exception as exc:  # pragma: no cover - protect outer flow
        print(f"❌ Unexpected error during diarization: {exc}")


def main() -> None:
    """CLI entry point."""
    run()


if __name__ == "__main__":
    main()
