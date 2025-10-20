"""Analysis tools for diarization results."""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class DiarizationAnalyzer:
    """Analyzes diarization results and provides statistics."""
    
    @staticmethod
    def analyze_diarization_result(diarization_result, return_data: bool = False) -> Optional[Dict]:
        """Analyze the diarization result to understand speaker distribution."""
        speakers = {}
        total_duration = 0
        total_audio_duration = 0
        
        # Get the total audio duration from the diarization result
        if hasattr(diarization_result, 'extent'):
            total_audio_duration = diarization_result.extent().duration
        else:
            # Fallback: calculate from the last segment
            for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
                total_audio_duration = max(total_audio_duration, turn.end)
        
        # Analyze speaker segments
        for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
            duration = turn.end - turn.start
            total_duration += duration
            
            if speaker_label not in speakers:
                speakers[speaker_label] = {'duration': 0, 'segments': 0}
            
            speakers[speaker_label]['duration'] += duration
            speakers[speaker_label]['segments'] += 1
        
        # Calculate silence/gap duration (this is normal!)
        silence_duration = total_audio_duration - total_duration
        
        # Create data structure for return
        diarization_data = {
            "total_audio_duration": total_audio_duration,
            "speech_duration": total_duration,
            "silence_duration": silence_duration,
            "silence_percentage": (silence_duration / total_audio_duration) * 100 if total_audio_duration > 0 else 0,
            "speakers": {}
        }
        
        for speaker, stats in speakers.items():
            percentage = (stats['duration'] / total_audio_duration) * 100 if total_audio_duration > 0 else 0
            avg_segment = stats['duration'] / stats['segments'] if stats['segments'] > 0 else 0
            
            diarization_data["speakers"][speaker] = {
                "duration": stats['duration'],
                "percentage": percentage,
                "segments": stats['segments'],
                "average_segment_length": avg_segment
            }
        
        # Print output 
        if not return_data:
            print(f"📊 Diarization Analysis:")
            print(f"   Total audio duration: {total_audio_duration:.1f}s")
            print(f"   Speech duration: {total_duration:.1f}s")
            if silence_duration > 0.1:
                print(f"   Silence/gaps: {silence_duration:.1f}s ({diarization_data['silence_percentage']:.1f}%)")
            print(f"   Found {len(speakers)} speakers:")
            
            for speaker, speaker_data in diarization_data["speakers"].items():
                print(f"   {speaker}: {speaker_data['duration']:.1f}s ({speaker_data['percentage']:.1f}%), "
                      f"{speaker_data['segments']} segments, avg {speaker_data['average_segment_length']:.1f}s")
        
        return diarization_data if return_data else speakers


class SegmentAnalyzer:
    """Analyzes final segments after speaker assignment."""
    
    @staticmethod
    def analyze_final_segments(segments: List[Dict]) -> Dict[str, Dict]:
        """Analyze the final segments after speaker assignment to find real UNKNOWN issues."""
        speaker_stats = {}
        total_duration = 0
        
        for segment in segments:
            speaker = segment.get('speaker', 'UNKNOWN')
            duration = segment['end'] - segment['start']
            total_duration += duration
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {'duration': 0, 'segments': 0, 'words': 0}
            
            speaker_stats[speaker]['duration'] += duration
            speaker_stats[speaker]['segments'] += 1
            speaker_stats[speaker]['words'] += len(segment.get('words', []))
        
        print()
        print(f"📊 Final Segment Analysis:")
        print(f"   Total transcribed duration: {total_duration:.1f}s")
        
        # Show all speakers including UNKNOWN
        for speaker, stats in sorted(speaker_stats.items()):
            percentage = (stats['duration'] / total_duration) * 100
            avg_segment = stats['duration'] / stats['segments']
            avg_words = stats['words'] / stats['segments'] if stats['segments'] > 0 else 0
            
            if speaker == "UNKNOWN":
                print(f"   🔍 {speaker}: {stats['duration']:.1f}s ({percentage:.1f}%), "
                      f"{stats['segments']} segments, {stats['words']} words "
                      f"(avg {avg_segment:.1f}s/seg, {avg_words:.1f} words/seg)")
            else:
                print(f"   {speaker}: {stats['duration']:.1f}s ({percentage:.1f}%), "
                      f"{stats['segments']} segments, {stats['words']} words")
        
        return speaker_stats


class BoundaryAnalyzer:
    """Analyzes speaker boundary issues and transitions."""
    
    @staticmethod
    def analyze_boundary_issues(segments: List[Dict], diarization_result, return_data: bool = False) -> Optional[Dict]:
        """
        Analyze where alignment issues occur.
        """
        unknown_count = sum(1 for seg in segments if seg.get('speaker') == 'UNKNOWN')
        unknown_percentage = (unknown_count / len(segments)) * 100 if len(segments) > 0 else 0
        
        # Find speaker changes
        speaker_changes = []
        for i in range(1, len(segments)):
            if segments[i]['speaker'] != segments[i-1]['speaker']:
                prev_seg = segments[i-1]
                curr_seg = segments[i]
                gap = curr_seg['start'] - prev_seg['end']
                
                change_data = {
                    "time": curr_seg['start'],
                    "from_speaker": prev_seg['speaker'],
                    "to_speaker": curr_seg['speaker'],
                    "gap_seconds": gap
                }
                speaker_changes.append(change_data)
        
        boundary_data = {
            "unknown_segments": unknown_count,
            "total_segments": len(segments),
            "unknown_percentage": unknown_percentage,
            "speaker_changes": speaker_changes,
            "total_speaker_changes": len(speaker_changes)
        }
        
        # Print output (existing behavior)
        if not return_data:
            print("\n🔍 Boundary Analysis:")
            print(f"   UNKNOWN segments: {unknown_count}/{len(segments)} ({unknown_percentage:.1f}%)")
            
            for change in speaker_changes:
                print(f"   Speaker change at {change['time']:.1f}s: {change['from_speaker']} -> {change['to_speaker']} (gap: {change['gap_seconds']:.1f}s)")
            
            print(f"   Total speaker changes: {len(speaker_changes)}")
        
        return boundary_data if return_data else None


class StatsExporter:
    """Handles exporting comprehensive analysis statistics."""
    
    @staticmethod
    def save_analysis_stats(segments: List[Dict], diarization_result, word_stats: Dict, segment_stats: Dict, 
                           speaker_stats: Dict, boundary_stats: Dict, settings: Dict, output_path: str) -> Dict:
        """
        Save comprehensive analysis statistics and settings to a JSON file.
        """
        # Collect diarization stats
        diarization_stats = DiarizationAnalyzer.analyze_diarization_result(diarization_result, return_data=True)
        
        # Get current timestamp
        log_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create comprehensive stats dictionary
        stats_data = {
            "summary": {
                "total_segments": len(segments),
                "total_duration_seconds": sum(seg['end'] - seg['start'] for seg in segments),
                "speakers_found": len([s for s in speaker_stats.keys() if s != "UNKNOWN"]),
                "unknown_percentage": (speaker_stats.get("UNKNOWN", {}).get("duration", 0) / 
                                     sum(stats["duration"] for stats in speaker_stats.values())) * 100 if speaker_stats else 0,
                "word_assignment_rate": (word_stats['assigned'] / word_stats['total']) * 100 if word_stats['total'] > 0 else 0,
            },
            "timestamp": log_timestamp,
            "settings": settings,
            "diarization_analysis": diarization_stats,
            "word_level_stats": word_stats,
            "segment_level_stats": segment_stats,
            "final_speaker_stats": speaker_stats,
            "boundary_analysis": boundary_stats
        }
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        return stats_data
