"""Speaker diarization pipeline and core processing logic."""

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
from pyannote.audio import Pipeline

from .config import DiarizationConfig
from .utils import DeviceManager


class DiarizationPipeline:
    """Manages the pyannote diarization pipeline."""
    
    def __init__(self, config: DiarizationConfig):
        self.config = config
        self.pipeline: Optional[Pipeline] = None
    
    def _safe_pipeline_cleanup(self, pipeline) -> None:
        """Safely clean up the diarization pipeline."""
        try:
            if pipeline:
                # Move to CPU first - use torch.device for proper type
                if hasattr(pipeline, 'to'):
                    pipeline.to(torch.device('cpu'))
                
                # Clear any internal caches
                if hasattr(pipeline, '_models'):
                    for model in pipeline._models.values():
                        if hasattr(model, 'to'):
                            model.to(torch.device('cpu'))
                
                del pipeline
                DeviceManager.clear_device_memory()
                
        except Exception as e:
            print(f"Warning: Error during pipeline cleanup: {e}")
    
    @contextmanager
    def safe_diarization_pipeline(self):
        """Context manager for safe pipeline handling."""
        pipeline = None
        
        try:
            # Load pipeline quietly for parameter testing
            is_parameter_testing = getattr(self.config, 'enable_parameter_testing', False)
            
            if not is_parameter_testing:
                print("  -> Loading speaker diarization models...")
            
            pipeline = Pipeline.from_pretrained(
                self.config.pipeline_model,
                use_auth_token=self.config.hugging_face_token
            )
            
            if not is_parameter_testing:
                print("  -> Configuring pipeline parameters...")
            
            # Configure parameters - suppress detailed output during parameter testing
            self._configure_pipeline_parameters(pipeline, verbose=not is_parameter_testing)
            
            if not is_parameter_testing:
                print(f"  -> Moving models to {self.config.device}...")
            
            pipeline.to(torch.device(self.config.device))
            
            if not is_parameter_testing:
                print("  -> Diarization pipeline ready!")
            
            yield pipeline
            
        except Exception as e:
            print(f"Error with diarization pipeline: {e}")
            raise
        finally:
            if pipeline:
                self._safe_pipeline_cleanup(pipeline)
    
    def diarize(self, audio_path: str) -> Any:
        """Perform diarization on an audio file."""
        with self.safe_diarization_pipeline() as pipeline:
            # Add progress indicator for the actual diarization processing
            is_parameter_testing = getattr(self.config, 'enable_parameter_testing', False)
            if not is_parameter_testing:
                print("  -> Running speaker diarization analysis...")
            
            result = pipeline(
                audio_path, 
                min_speakers=self.config.min_speakers, 
                max_speakers=self.config.max_speakers
            )
            
            if not is_parameter_testing:
                print("  -> Diarization analysis complete!")
            
            return result

    def _configure_pipeline_parameters(self, pipeline, verbose: bool = True):
        """Configure pipeline parameters safely."""
        try:
            # Configure segmentation if available
            if hasattr(pipeline, '_segmentation'):
                seg_config = {
                    'onset': self.config.segmentation_threshold,
                    'offset': self.config.segmentation_threshold,
                    'min_duration_on': self.config.min_duration_on,
                    'min_duration_off': self.config.min_duration_off
                }
                
                configured_params = []
                for param, value in seg_config.items():
                    if hasattr(pipeline._segmentation, param):
                        setattr(pipeline._segmentation, param, value)
                        configured_params.append(param)
                
                if verbose and configured_params:
                    print(f"  Segmentation parameters configured: {configured_params}")
            
            # Configure clustering if available
            if hasattr(pipeline, '_clustering'):
                clustering_params = []
                
                if hasattr(pipeline._clustering, 'threshold'):
                    pipeline._clustering.threshold = self.config.clustering_threshold
                    clustering_params.append('threshold')
                
                if hasattr(pipeline._clustering, 'min_cluster_size'):
                    pipeline._clustering.min_cluster_size = self.config.clustering_min_cluster_size
                    clustering_params.append('min_cluster_size')
                
                if hasattr(pipeline._clustering, 'method'):
                    pipeline._clustering.method = self.config.clustering_method
                    clustering_params.append('method')
                
                if verbose and clustering_params:
                    print(f"  Clustering parameters configured: {clustering_params}")
            
            # Configure embedding distance threshold if available
            if hasattr(self.config, 'embedding_distance_threshold'):
                embedding_configured = False
                
                # Try different possible locations for embedding threshold
                if hasattr(pipeline, '_embedding') and hasattr(pipeline._embedding, 'threshold'):
                    pipeline._embedding.threshold = self.config.embedding_distance_threshold
                    embedding_configured = True
                elif hasattr(pipeline, '_clustering') and hasattr(pipeline._clustering, 'distance_threshold'):
                    pipeline._clustering.distance_threshold = self.config.embedding_distance_threshold
                    embedding_configured = True
                elif hasattr(pipeline, '_clustering') and hasattr(pipeline._clustering, 'embedding_threshold'):
                    pipeline._clustering.embedding_threshold = self.config.embedding_distance_threshold
                    embedding_configured = True
                
                if verbose and embedding_configured:
                    print("  Embedding distance threshold configured.")
            
            if verbose:
                print("  Pipeline configured with custom parameters.")
                
        except Exception as config_error:
            if verbose:
                print(f"  Configuration note: Some parameters use defaults ({config_error})")

class SpeakerAligner:
    """Aligns transcription segments with speaker diarization results."""
    
    def __init__(self, config: DiarizationConfig):
        self.config = config
    
    def align_segments_with_speakers(self, whisper_result: Dict, diarization_result, gap_log_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Align Whisper segments with diarization results at word level.
        
        Returns:
            Dictionary containing aligned segments and statistics
        """
        from .utils import SpeakerAssigner, SilenceMarkerProcessor, WordProcessor
        
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Aligning transcription with speaker segments (word-level)...")
        
        # Process each segment
        final_segments = []
        word_stats = {'total': 0, 'unknown': 0, 'assigned': 0}
        segment_stats = {'total': 0, 'high_conf': 0, 'low_conf': 0, 'unknown': 0}
        
        for segment in whisper_result["segments"]:
            # First, assign speakers to individual words
            for word in segment.get('words', []):
                word_start = word.get('start', segment['start'])
                word_end = word.get('end', segment['end'])
                
                word_speaker = SpeakerAssigner.find_speaker_for_word(word_start, word_end, diarization_result)
                word['speaker'] = word_speaker
                
                # Track statistics
                word_stats['total'] += 1
                if word_speaker == "UNKNOWN":
                    word_stats['unknown'] += 1
                else:
                    word_stats['assigned'] += 1
            
            # Then assign segment speaker based on word majority
            segment_speaker, confidence = SpeakerAssigner.assign_segment_speaker_from_words(segment)
            segment['speaker'] = segment_speaker
            segment['speaker_confidence'] = confidence
            
            # Track segment statistics
            segment_stats['total'] += 1
            if segment_speaker == "UNKNOWN":
                segment_stats['unknown'] += 1
            elif confidence > 0.7:
                segment_stats['high_conf'] += 1
            else:
                segment_stats['low_conf'] += 1
            
            final_segments.append(segment)
        
        # Update the result
        whisper_result['segments'] = final_segments
        
        # Print detailed statistics
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Word-level alignment complete.") 
        print(f"   Words: {word_stats['assigned']}/{word_stats['total']} assigned ({word_stats['assigned']/word_stats['total']*100:.1f}%)")
        print(f"   Segments: High conf: {segment_stats['high_conf']}, Low conf: {segment_stats['low_conf']}, Unknown: {segment_stats['unknown']}")
        
        # Apply smoothing if enabled
        if self.config.smoothing_enabled:
            print("Applying word-level transition smoothing...")
            whisper_result['segments'] = SpeakerAssigner.smooth_word_level_transitions(
                whisper_result['segments'], 
                self.config.min_speaker_words
            )

        if not self.config.preserve_markers:
            whisper_result['segments'] = WordProcessor.remove_disfluency_markers_from_segments(
                whisper_result['segments']
            )
        
        # Add silence markers if enabled
        if self.config.include_silence_markers:
            print("Adding silence markers...")
            whisper_result['segments'] = SilenceMarkerProcessor.add_word_level_silence_markers(
                whisper_result['segments'], 
                self.config.min_silence_duration,
                gap_log_path=gap_log_path
            )
            word_silences = sum(len([w for w in seg.get('words', []) if w.get('is_silence_marker', False)]) 
                             for seg in whisper_result['segments'])
            print(f"Added {word_silences} silence markers")
            if gap_log_path and os.path.exists(gap_log_path):
                relative_gap_path = os.path.relpath(gap_log_path, self.config.final_output_dir)
                print(f"Silence gap durations logged to {relative_gap_path}")
        
        return {
            'segments': whisper_result['segments'],
            'word_stats': word_stats,
            'segment_stats': segment_stats
        }
