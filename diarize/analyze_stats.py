"""
Stats Analysis Tool for Diarization Results
Analyzes JSON stats files and creates overview tables.
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class StatsMetrics:
    """Container for key diarization metrics."""
    num_speakers: int
    num_segments: int
    avg_segment_length: float
    speaker_word_percentages: Dict[str, float]
    total_words: int
    speaker_turns: Dict[str, int]
    avg_words_per_segment: float
    silence_ratio: Optional[float] = None
    longest_segment: Optional[float] = None
    shortest_segment: Optional[float] = None

class DiarizationStatsAnalyzer:
    """Analyzes diarization statistics from JSON files."""
    
    def __init__(self, config):
        self.config = config
        self.stats_dir = os.path.join(config.final_output_dir, 'stats')
        
    def scan_stats_files(self) -> Dict[str, List[str]]:
        """Scan stats directory and group files by audio source."""
        if not os.path.exists(self.stats_dir):
            print(f"Stats directory not found: {self.stats_dir}")
            return {}
            
        stats_files = {}
        for file in Path(self.stats_dir).glob('*.json'):
            # Extract base filename (remove parameter suffixes)
            base_name = self._extract_base_name(file.stem)
            if base_name not in stats_files:
                stats_files[base_name] = []
            stats_files[base_name].append(str(file))
            
        return stats_files
    
    def _extract_base_name(self, filename: str) -> str:
        """Extract base audio filename from stats filename."""
        # Remove common parameter suffixes
        suffixes_to_remove = [
            '_embed', '_cluster', '_seg', '_stats',
            '_test', '_param'
        ]
        base = filename
        for suffix in suffixes_to_remove:
            if suffix in base:
                base = base.split(suffix)[0]
        return base
    
    def load_stats_file(self, filepath: str) -> Optional[Dict]:
        """Load and parse a stats JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def extract_metrics(self, stats_data: Dict) -> StatsMetrics:
        """Extract key metrics from stats data."""
        # Get segments summary
        segments_summary = stats_data.get('segments_summary', {})
        metadata = stats_data.get('metadata', {})
        statistics = stats_data.get('statistics', {})
        
        # Basic counts
        num_speakers = segments_summary.get('speakers_detected', 0)
        num_segments = segments_summary.get('total_segments', 0)
        total_duration = segments_summary.get('total_duration', 0)
        
        # Speaker segment counts
        segments_by_speaker = segments_summary.get('segments_by_speaker', {})
        
        # Calculate speaker percentages by segment count (approximation)
        total_speaker_segments = sum(count for speaker, count in segments_by_speaker.items() 
                                   if speaker not in ['UNKNOWN'])
        
        speaker_word_percentages = {}
        speaker_turns = {}
        
        for speaker, segment_count in segments_by_speaker.items():
            if speaker != 'UNKNOWN':
                # Approximate word percentage by segment percentage
                speaker_word_percentages[speaker] = (segment_count / total_speaker_segments * 100) if total_speaker_segments > 0 else 0
                speaker_turns[speaker] = segment_count
        
        # Calculate average segment length
        avg_segment_length = total_duration / num_segments if num_segments > 0 else 0
        
        # Estimate total words (rough approximation: 2-3 words per second of speech)
        estimated_words_per_second = 2.5
        total_words = int(total_duration * estimated_words_per_second)
        
        # Calculate average words per segment
        avg_words_per_segment = total_words / num_segments if num_segments > 0 else 0
        
        # Get speaker change statistics
        total_speaker_changes = statistics.get('total_speaker_changes', 0)
        
        return StatsMetrics(
            num_speakers=num_speakers,
            num_segments=num_segments,
            avg_segment_length=avg_segment_length,
            speaker_word_percentages=speaker_word_percentages,
            total_words=total_words,
            speaker_turns=speaker_turns,
            avg_words_per_segment=avg_words_per_segment,
            silence_ratio=None,  # Not available in current format
            longest_segment=None,  # Not directly available
            shortest_segment=None  # Not directly available
        )
    
    def create_comparison_table(self, audio_file: str, stats_files: List[str]) -> pd.DataFrame:
        """Create comparison table for a single audio file."""
        rows = []
        
        for stats_file in stats_files:
            stats_data = self.load_stats_file(stats_file)
            if not stats_data:
                continue
                
            metrics = self.extract_metrics(stats_data)
            
            # Extract parameter configuration from filename or stats
            params = self._extract_parameters(stats_file, stats_data)
            
            # Create row data
            row = {
                'File': Path(stats_file).stem,
                'Num_Speakers': metrics.num_speakers,
                'Num_Segments': metrics.num_segments,
                'Avg_Segment_Length': round(metrics.avg_segment_length, 2),
                'Estimated_Total_Words': metrics.total_words,
                'Avg_Words_Per_Segment': round(metrics.avg_words_per_segment, 1),
                'Total_Duration': round(stats_data.get('segments_summary', {}).get('total_duration', 0), 1),
                'Speaker_Changes': stats_data.get('statistics', {}).get('total_speaker_changes', 0),
            }
            
            # Add speaker distribution columns (by segment percentage)
            for i, (speaker, percentage) in enumerate(sorted(metrics.speaker_word_percentages.items())):
                row[f'Speaker_{i+1}_Segments_%'] = round(percentage, 1)
                row[f'Speaker_{i+1}_Segment_Count'] = metrics.speaker_turns.get(speaker, 0)
            
            # Add parameter columns
            param_cols = {}
            for key, value in params.items():
                if value is not None and value != 'N/A':
                    param_cols[key] = value
            row.update(param_cols)
            
            rows.append(row)
        
        if not rows:
            return pd.DataFrame()
            
        df = pd.DataFrame(rows)
        
        # Sort by parameter combinations for easier comparison
        param_cols = [col for col in df.columns if col.startswith(('embedding', 'clustering', 'segmentation'))]
        if param_cols:
            df = df.sort_values(param_cols)
        else:
            # Sort by number of speakers and then by segments
            df = df.sort_values(['Num_Speakers', 'Num_Segments'])
            
        return df
    
    def _extract_parameters(self, filepath: str, stats_data: Dict) -> Dict[str, Any]:
        """Extract parameter values from filename or stats data."""
        params = {}
        
        # Get configuration from stats data
        config_data = stats_data.get('configuration', {})
        
        if config_data:
            # Diarization settings
            diarization_config = config_data.get('diarization', {})
            if diarization_config:
                params['model'] = diarization_config.get('model', 'N/A')
                params['min_speakers'] = diarization_config.get('min_speakers')
                params['max_speakers'] = diarization_config.get('max_speakers')
                
                # Pipeline configuration
                pipeline_config = diarization_config.get('pipeline_config', {})
                if pipeline_config:
                    # Segmentation settings
                    seg_config = pipeline_config.get('segmentation', {})
                    params['segmentation_threshold'] = seg_config.get('threshold')
                    params['min_duration_on'] = seg_config.get('min_duration_on')
                    params['min_duration_off'] = seg_config.get('min_duration_off')
                    
                    # Clustering settings
                    cluster_config = pipeline_config.get('clustering', {})
                    params['clustering_method'] = cluster_config.get('method')
                    params['clustering_threshold'] = cluster_config.get('threshold')
                    params['clustering_min_cluster_size'] = cluster_config.get('min_cluster_size')
                    
                    # Advanced settings
                    params['embedding_distance_threshold'] = pipeline_config.get('embedding_distance_threshold')
            
            # Device settings
            params['device'] = config_data.get('device')
        
        # Add processing timestamp
        metadata = stats_data.get('metadata', {})
        params['processing_timestamp'] = metadata.get('processing_timestamp', 'unknown')
        
        return params
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report with one row per run."""
        stats_files = self.scan_stats_files()
        
        if not stats_files:
            return "No stats files found."
        
        report = ["# Diarization Stats Analysis Report\n"]
        report.append(f"**Stats Directory:** {self.stats_dir}\n")
        report.append(f"**Processing runs analyzed:** {sum(len(v) for v in stats_files.values())}\n")
        
        # Prepare all rows for a single table
        all_rows = []
        for audio_file, file_list in stats_files.items():
            for stats_file in file_list:
                stats_data = self.load_stats_file(stats_file)
                if not stats_data:
                    continue
                metrics = self.extract_metrics(stats_data)
                params = self._extract_parameters(stats_file, stats_data)
                row = {
                    'Audio_File': audio_file,
                    'Stats_File': os.path.basename(stats_file),
                    'Num_Speakers': metrics.num_speakers,
                    'Num_Segments': metrics.num_segments,
                    'Avg_Segment_Length': round(metrics.avg_segment_length, 2),
                    'Estimated_Total_Words': metrics.total_words,
                    'Avg_Words_Per_Segment': round(metrics.avg_words_per_segment, 1),
                    'Total_Duration': round(stats_data.get('segments_summary', {}).get('total_duration', 0), 1),
                    'Speaker_Changes': stats_data.get('statistics', {}).get('total_speaker_changes', 0),
                }
                # Add speaker distribution columns
                for i, (speaker, percentage) in enumerate(sorted(metrics.speaker_word_percentages.items())):
                    row[f'Speaker_{i+1}_Segments_%'] = round(percentage, 1)
                    row[f'Speaker_{i+1}_Segment_Count'] = metrics.speaker_turns.get(speaker, 0)
                # Add parameter columns
                for key, value in params.items():
                    if value is not None and value != 'N/A':
                        row[key] = value
                all_rows.append(row)
        if not all_rows:
            return "No valid data found."
        df = pd.DataFrame(all_rows)
        # Save to Excel and CSV
        excel_path = os.path.join(self.config.final_output_dir, 'diarization_analysis.xlsx')
        csv_path = os.path.join(self.config.final_output_dir, 'diarization_analysis.csv')
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='AllRuns', index=False)
            df.to_csv(csv_path, index=False)
            report.append(f"\n**Detailed analysis saved to:** {excel_path} (sheet: AllRuns)")
            report.append(f"\n**CSV also saved to:** {csv_path}")
        except Exception as e:
            print(f"Warning: Could not create Excel file: {e}")
            report.append(f"\n**Note:** Excel/CSV file could not be created: {e}")
        # Add summary stats to report
        report.append(f"\n**Unique audio files:** {len(stats_files)}")
        report.append(f"**Total processing runs:** {len(all_rows)}")
        report.append(f"**Columns:** {', '.join(df.columns)}")
        return '\n'.join(report)
    
    def _find_best_configuration(self, df: pd.DataFrame) -> Optional[Dict]:
        """Find the best configuration based on heuristics."""
        if df.empty:
            return None
        
        try:
            # Simple heuristic: prefer 2 speakers with balanced distribution
            target_speakers = 2
            speaker_filtered = df[df['Num_Speakers'] == target_speakers]
            
            if speaker_filtered.empty:
                speaker_filtered = df
            
            # Look for most balanced speaker distribution
            speaker_cols = [col for col in speaker_filtered.columns if col.startswith('Speaker_') and col.endswith('_Segments_%')]
            
            if len(speaker_cols) >= 2:
                # Calculate balance score for 2-speaker scenarios
                first_col = speaker_cols[0]
                second_col = speaker_cols[1]
                speaker_filtered = speaker_filtered.copy()
                speaker_filtered['balance_score'] = abs(speaker_filtered[first_col] - 50) + abs(speaker_filtered[second_col] - 50)
                # Remove rows with NaN balance scores
                speaker_filtered = speaker_filtered.dropna(subset=['balance_score'])
                if not speaker_filtered.empty:
                    best_row = speaker_filtered.loc[speaker_filtered['balance_score'].idxmin()]
                else:
                    best_row = df.iloc[0]
            else:
                # Fallback: choose configuration with fewest speaker changes relative to segments
                if 'Speaker_Changes' in speaker_filtered.columns and 'Num_Segments' in speaker_filtered.columns:
                    speaker_filtered = speaker_filtered.copy()
                    # Avoid division by zero
                    mask = speaker_filtered['Num_Segments'] > 0
                    speaker_filtered = speaker_filtered[mask]
                    if not speaker_filtered.empty:
                        speaker_filtered['change_ratio'] = speaker_filtered['Speaker_Changes'] / speaker_filtered['Num_Segments']
                        # Remove rows with NaN change ratios
                        speaker_filtered = speaker_filtered.dropna(subset=['change_ratio'])
                        if not speaker_filtered.empty:
                            best_row = speaker_filtered.loc[speaker_filtered['change_ratio'].idxmin()]
                        else:
                            best_row = df.iloc[0]
                    else:
                        best_row = df.iloc[0]
                else:
                    best_row = speaker_filtered.iloc[0]
            
            config_parts = []
            if 'embedding_distance_threshold' in best_row.index and pd.notna(best_row['embedding_distance_threshold']):
                config_parts.append(f"embed_{best_row['embedding_distance_threshold']}")
            if 'clustering_threshold' in best_row.index and pd.notna(best_row['clustering_threshold']):
                config_parts.append(f"cluster_{best_row['clustering_threshold']}")
            if 'segmentation_threshold' in best_row.index and pd.notna(best_row['segmentation_threshold']):
                config_parts.append(f"seg_{best_row['segmentation_threshold']}")
            
            config_summary = "_".join(config_parts) if config_parts else "default_config"
            
            return {
                'config_summary': config_summary,
                'speakers': best_row['Num_Speakers'],
                'segments': best_row['Num_Segments'],
                'balance_score': best_row.get('balance_score', 'N/A'),
                'speaker_changes': best_row.get('Speaker_Changes', 'N/A')
            }
            
        except Exception as e:
            print(f"Warning: Error in best configuration analysis: {e}")
            # Return basic info from first row
            if not df.empty:
                first_row = df.iloc[0]
                return {
                    'config_summary': 'analysis_error',
                    'speakers': first_row['Num_Speakers'],
                    'segments': first_row['Num_Segments'],
                    'balance_score': 'N/A',
                    'speaker_changes': first_row.get('Speaker_Changes', 'N/A')
                }
            return None

def main():
    """Main function to run the stats analysis."""
    from config import DEFAULT_DIARIZATION_CONFIG
    
    analyzer = DiarizationStatsAnalyzer(DEFAULT_DIARIZATION_CONFIG)
    
    print("Scanning stats files...")
    stats_files = analyzer.scan_stats_files()
    
    if not stats_files:
        print("No stats files found!")
        return
    
    print(f"Found stats for {len(stats_files)} audio files")
    
    # Generate comprehensive report
    report = analyzer.generate_summary_report()
    
    # Save report
    report_path = os.path.join(DEFAULT_DIARIZATION_CONFIG.final_output_dir, 'analysis_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nAnalysis complete!")
    print(f"Report saved to: {report_path}")
    
    # Check if Excel was created or CSV fallback used
    excel_path = os.path.join(DEFAULT_DIARIZATION_CONFIG.final_output_dir, 'diarization_analysis.xlsx')
    csv_dir = os.path.join(DEFAULT_DIARIZATION_CONFIG.final_output_dir, 'csv_analysis')
    
    if os.path.exists(excel_path):
        print(f"Excel analysis saved to: {excel_path}")
    elif os.path.exists(csv_dir):
        print(f"CSV analysis files saved to: {csv_dir}")
    else:
        print("Analysis data files could not be created")
    
    # Print summary to console
    print("\n" + "="*50)
    print(report)

if __name__ == "__main__":
    main()