"""Package initialization for diarization modules."""

from .config import DiarizationConfig, DEFAULT_DIARIZATION_CONFIG
from .pipeline import DiarizationPipeline, SpeakerAligner
from .analysis import DiarizationAnalyzer, SegmentAnalyzer, BoundaryAnalyzer
from .output import (
    TranscriptFormatter, VTTFormatter, RTTMFormatter, 
    RTFFormatter, TXTFormatter, TSVFormatter, PyannoteSegmentFormatter, StatsExporter
)
from .utils import (
    DeviceManager, LoggerManager, SilenceMarkerProcessor,
    WordProcessor, SpeakerAssigner, BackfillTranscriber
)

# Import the main function from the diarize.py script
import importlib.util
import os

def main():
    """Import and execute the main function from diarize.py"""
    # Get the path to diarize.py in the parent directory
    diarize_py_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'diarize.py')
    
    # Load the diarize.py module
    spec = importlib.util.spec_from_file_location("diarize_main", diarize_py_path)
    diarize_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diarize_module)
    
    # Call the main function from diarize.py
    return diarize_module.main()

__all__ = [
    'main',
    'DiarizationConfig',
    'DEFAULT_DIARIZATION_CONFIG',
    'DiarizationPipeline',
    'SpeakerAligner',
    'DiarizationAnalyzer',
    'SegmentAnalyzer', 
    'BoundaryAnalyzer',
    'TranscriptFormatter',
    'VTTFormatter',
    'RTTMFormatter',
    'RTFFormatter',
    'TXTFormatter',
    'TSVFormatter',
    'PyannoteSegmentFormatter',
    'StatsExporter',
    'DeviceManager',
    'LoggerManager',
    'SilenceMarkerProcessor',
    'WordProcessor',
    'SpeakerAssigner',
    'BackfillTranscriber'
]
