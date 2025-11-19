"""SenseVoice provider for enhanced speech recognition with emotion and event detection."""

import numpy as np
from typing import Optional, Dict, Any, List
from .utils import DeviceManager


class SenseVoiceProvider:
    """Wrapper for SenseVoice model inference.
    
    Provides speech recognition with additional emotion recognition and audio event detection.
    """

    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: Optional[str] = None,
        use_vad: bool = False,
    ):
        """Initialize SenseVoice provider.
        
        Args:
            model_name: Model identifier (e.g., "iic/SenseVoiceSmall")
            device: Device to use for inference (e.g., "cuda:0", "cpu")
            use_vad: Whether to use VAD (Voice Activity Detection) - typically False for short backfill snippets
        """
        self.model_name = model_name
        # Default to mps on macOS when no device provided; fall back to cpu otherwise
        self.device = device or "mps"
        self.use_vad = use_vad
        self._model = None
        self._model_kwargs = {}
        
    def _ensure_model_loaded(self) -> None:
        """Lazy load the SenseVoice model."""
        if self._model is not None:
            return
            
        try:
            from funasr import AutoModel
            
            print(f"      Loading SenseVoice model: {self.model_name}")
            
            # Build model configuration
            model_kwargs = {
                "model": self.model_name,
                "trust_remote_code": True,
                "device": self.device,
                # Prevent funasr/ModelScope from checking for updates every load
                # (funasr AutoModel supports disable_update=True as of funasr>=1.2.x)
                "disable_update": True,
            }
            
            # Add VAD if requested (typically not needed for short backfill snippets)
            if self.use_vad:
                model_kwargs["vad_model"] = "fsmn-vad"
                model_kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
            
            # Try to initialize AutoModel with disable_update if supported.
            try:
                self._model = AutoModel(**model_kwargs)
                used_disable_update = True
            except TypeError:
                # Older/newer versions of AutoModel may not accept disable_update; retry without it
                if 'disable_update' in model_kwargs:
                    model_kwargs.pop('disable_update')
                self._model = AutoModel(**model_kwargs)
                used_disable_update = False

            # store any kwargs exposed by the underlying model
            self._model_kwargs = self._model.kwargs if hasattr(self._model, 'kwargs') else {}

            if used_disable_update:
                print(f"      SenseVoice model loaded successfully (disable_update=True)")
            else:
                print(f"      SenseVoice model loaded successfully")
            
        except ImportError as exc:
            raise RuntimeError(
                "funasr package is required for SenseVoice. "
                "Install with: pip install 'funasr>=1.0.0' modelscope"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load SenseVoice model: {exc}") from exc
    
    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "auto",
        use_itn: bool = False,
    ) -> Dict[str, Any]:
        """Transcribe audio with SenseVoice.
        
        Args:
            audio: Audio data as numpy array (16kHz, mono, float32)
            language: Language code ("auto", "zh", "en", "yue", "ja", "ko", "nospeech")
            use_itn: Whether to use Inverse Text Normalization
            
        Returns:
            Dictionary containing:
                - text: Transcribed text (cleaned)
                - raw_text: Raw text with emotion/event tags
                - emotion: Detected emotion tag (e.g., "<|HAPPY|>", "<|NEUTRAL|>")
                - event: Detected event tag (e.g., "<|Speech|>", "<|Laughter|>")
                - words: List of word-level timing information (if available)
        """
        self._ensure_model_loaded()
        
        try:
            # SenseVoice expects audio as numpy array (already 16kHz mono)
            result = self._model.generate(
                input=audio,
                cache={},
                language=language,
                use_itn=use_itn,
                batch_size=1,
            )
            
            if not result or len(result) == 0:
                return self._empty_result()
            
            # Extract first result
            result_data = result[0] if isinstance(result, list) else result
            
            # Parse the result
            return self._parse_sensevoice_result(result_data)
            
        except Exception as exc:
            print(f"      Warning: SenseVoice transcription failed: {exc}")
            return self._empty_result()
    
    def _parse_sensevoice_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse SenseVoice output and extract components.
        
        SenseVoice returns text with embedded tags like:
        "<|zh|><|NEUTRAL|><|Speech|><|woitn|>transcribed text here"
        """
        raw_text = result.get("text", "")
        
        # Extract emotion and event tags using the postprocessing utility
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
            clean_text = rich_transcription_postprocess(raw_text)
        except ImportError:
            # Fallback: simple tag removal if postprocess utility not available
            clean_text = self._simple_tag_removal(raw_text)
        
        # Extract tags from raw text
        emotion = self._extract_tag(raw_text, ["HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"])
        event = self._extract_tag(raw_text, ["Speech", "Laughter", "Applause", "Crying", "Coughing", "Sneezing", "BGM"])
        language = self._extract_tag(raw_text, ["zh", "en", "yue", "ja", "ko", "nospeech"])
        
        # Word-level timestamps (if available in result)
        words = result.get("words", [])
        
        return {
            "text": clean_text.strip(),
            "raw_text": raw_text,
            "emotion": emotion,
            "event": event,
            "language": language,
            "words": words,
        }
    
    def _extract_tag(self, text: str, tag_options: List[str]) -> Optional[str]:
        """Extract a specific tag type from SenseVoice output.
        
        Args:
            text: Raw text with tags
            tag_options: List of possible tag values to search for
            
        Returns:
            The tag found (e.g., "<|HAPPY|>") or None
        """
        for tag in tag_options:
            tag_pattern = f"<|{tag}|>"
            if tag_pattern in text:
                return tag_pattern
        return None
    
    def _simple_tag_removal(self, text: str) -> str:
        """Simple fallback method to remove tags from text."""
        import re
        # Remove patterns like <|tag|>
        cleaned = re.sub(r'<\|[^|]+\|>', '', text)
        return cleaned.strip()
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "text": "",
            "raw_text": "",
            "emotion": None,
            "event": None,
            "language": None,
            "words": [],
        }
    
    def close(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_kwargs = {}
            DeviceManager.clear_device_memory()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        return False
