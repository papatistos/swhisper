#!/usr/bin/env python3
"""
Quick viewer for diarization analysis results.
Shows summary and detailed analysis of parameter test results.
"""

import pandas as pd
import os
from config import DEFAULT_DIARIZATION_CONFIG

def view_analysis():
    """Display analysis results in a readable format."""
    
    # Paths
    excel_path = os.path.join(DEFAULT_DIARIZATION_CONFIG.final_output_dir, 'diarization_analysis.xlsx')
    report_path = os.path.join(DEFAULT_DIARIZATION_CONFIG.final_output_dir, 'analysis_report.md')
    
    print("="*80)
    print("DIARIZATION ANALYSIS VIEWER")
    print("="*80)
    
    # Check if files exist
    if not os.path.exists(excel_path):
        print(f"❌ Excel file not found at: {excel_path}")
        print("💡 Run 'python analyze_stats.py' first to generate the analysis.")
        return
    
    try:
        # Read Excel file
        xl_file = pd.ExcelFile(excel_path)
        sheet_names = xl_file.sheet_names
        
        print(f"📊 Analysis file: {excel_path}")
        print(f"📋 Total sheets: {len(sheet_names)}")
        
        # Separate data sheets from summary
        data_sheets = [s for s in sheet_names if s != 'Summary']
        print(f"📁 Audio file analyses: {len(data_sheets)}")
        print()
        
        # Show summary if available
        if 'Summary' in sheet_names:
            print("🎯 SUMMARY TABLE:")
            print("-" * 50)
            summary_df = pd.read_excel(excel_path, sheet_name='Summary')
            
            # Display with better formatting
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', 40)
            
            print(summary_df.to_string(index=False))
            print()
            
            # Show some statistics
            print("📈 QUICK INSIGHTS:")
            print("-" * 50)
            
            # Count files by speaker range
            speaker_counts = {}
            for _, row in summary_df.iterrows():
                speaker_range = row['Speaker_Range']
                speaker_counts[speaker_range] = speaker_counts.get(speaker_range, 0) + 1
            
            print("Speaker detection distribution:")
            for speaker_range, count in sorted(speaker_counts.items()):
                print(f"  • {speaker_range} speakers: {count} processing runs")
            
            # Show most common configurations
            print("\nMost common configurations:")
            config_counts = summary_df['Best_Config'].value_counts().head(5)
            for config, count in config_counts.items():
                print(f"  • {config}: {count} runs")
            
            print()
        
        # Group sheets by base audio file name
        print("📋 DETAILED ANALYSIS BY AUDIO FILE:")
        print("-" * 50)
        
        # Group sheets by base name
        base_files = {}
        for sheet_name in data_sheets:
            # Extract base name (remove timestamp)
            base_name = sheet_name.split('_202')[0]  # Split at timestamp
            if base_name not in base_files:
                base_files[base_name] = []
            base_files[base_name].append(sheet_name)
        
        for base_name, sheets in base_files.items():
            print(f"\n🎵 {base_name}")
            print(f"   Processing runs: {len(sheets)}")
            
            # Show data from each processing run
            for i, sheet_name in enumerate(sheets):
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                
                if not df.empty:
                    row = df.iloc[0]  # Each sheet has only 1 row
                    
                    print(f"   Run {i+1}: {sheet_name}")
                    print(f"      • Speakers: {row['Num_Speakers']}")
                    print(f"      • Segments: {row['Num_Segments']}")
                    print(f"      • Avg segment: {row['Avg_Segment_Length']:.2f}s")
                    print(f"      • Duration: {row['Total_Duration']:.1f}s")
                    
                    # Show parameter configuration
                    config_parts = []
                    if pd.notna(row.get('embedding_distance_threshold')):
                        config_parts.append(f"embed={row['embedding_distance_threshold']}")
                    if pd.notna(row.get('clustering_threshold')):
                        config_parts.append(f"cluster={row['clustering_threshold']}")
                    if pd.notna(row.get('segmentation_threshold')):
                        config_parts.append(f"seg={row['segmentation_threshold']}")
                    
                    if config_parts:
                        print(f"      • Config: {', '.join(config_parts)}")
                    
                    # Show speaker distribution
                    speaker_info = []
                    for col in df.columns:
                        if col.startswith('Speaker_') and col.endswith('_Segments_%'):
                            speaker_num = col.split('_')[1]
                            percentage = row[col]
                            if pd.notna(percentage) and percentage > 0:
                                speaker_info.append(f"Spk{speaker_num}: {percentage:.1f}%")
                    
                    if speaker_info:
                        print(f"      • Distribution: {', '.join(speaker_info)}")
                    
                    print()
                else:
                    print(f"   {sheet_name}: No data available")
        
        print("="*80)
        print("📝 PROCESSING RUNS SUMMARY:")
        print("="*80)
        
        # Count base audio files
        print(f"📁 Unique audio files analyzed: {len(base_files)}")
        print(f"🔄 Total processing runs: {len(data_sheets)}")
        print()
        
        # Show which files have multiple runs
        for base_name, sheets in base_files.items():
            if len(sheets) > 1:
                print(f"� {base_name}: {len(sheets)} different processing runs")
                
                # Compare configurations if multiple runs
                configs = []
                for sheet_name in sheets:
                    df = pd.read_excel(excel_path, sheet_name=sheet_name)
                    if not df.empty:
                        row = df.iloc[0]
                        config = []
                        if pd.notna(row.get('embedding_distance_threshold')):
                            config.append(f"e{row['embedding_distance_threshold']}")
                        if pd.notna(row.get('clustering_threshold')):
                            config.append(f"c{row['clustering_threshold']}")
                        configs.append("_".join(config) if config else "default")
                
                if len(set(configs)) > 1:
                    print(f"   Different configurations tested: {', '.join(set(configs))}")
                else:
                    print(f"   Same configuration: {configs[0] if configs else 'unknown'}")
            else:
                print(f"📄 {base_name}: 1 processing run")
        
        print()
        
        # Show markdown report if available (truncated)
        if os.path.exists(report_path):
            print("📝 MARKDOWN REPORT (first 1000 chars):")
            print("-" * 50)
            with open(report_path, 'r') as f:
                content = f.read()
                if len(content) > 1000:
                    print(content[:1000] + "...")
                    print(f"\n💡 Full report: {report_path}")
                else:
                    print(content)
        else:
            print(f"❌ Report file not found at: {report_path}")
            
    except Exception as e:
        print(f"❌ Error reading analysis files: {e}")
        print("💡 Try running 'python analyze_stats.py' to regenerate the analysis.")

if __name__ == "__main__":
    view_analysis()
