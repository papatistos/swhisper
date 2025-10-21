# Diarization Stats Analysis Tools

This set of very rudimentary tools helps you analyze the results from your diarization parameter testing by scanning JSON stats files and creating comprehensive overview tables.

## 📁 Files

- `analyze_stats.py` - Main analysis script that processes JSON stats files
- `view_analysis.py` - Viewer script for formatted display of results  
- `config.py` - Configuration file (already existing)

## 🚀 Usage

To produce multiple diarizations with different parameter settings, set `enable_parameter_testing` to `True` in `diarize/config.py` and adjust the parameter ranges as desired. After running diarization, use the analysis tool as described below. ALternatively, you can use it on any manuallully produced stats files.

### 1. Generate Analysis

```bash
python diarize/analyze_stats.py
```

This will:
- Scan the `transcripts/stats/` directory (under the active path settings) for JSON files
- Extract key metrics from each stats file
- Create `diarization_analysis.xlsx` with a single `AllRuns` sheet listing one row per processing run
- Write a matching CSV (`diarization_analysis.csv`) and a markdown summary report (`analysis_report.md`)

### 2. View Results

```bash
python view_analysis.py
```

## 📊 Output Files

The analysis generates several output files in the transcripts directory:

- `diarization_analysis.xlsx` (`AllRuns` sheet consolidates every run)
- `diarization_analysis.csv` (same data as the Excel sheet)
- `analysis_report.md` (high-level markdown summary)


## 📈 Metrics Analyzed

For each parameter combination, the analysis extracts:

### Key Statistics
- Number of speakers detected
- Number of segments 
- Average segment duration
- Average word count per segment
- Average words per segment
- Total duration of audio

### Speaker Distribution 
- Speaker segment counts
- Speaker segment percentages

### Configuration Parameters
- Embedding distance threshold
- Clustering threshold 
- Segmentation threshold
- Model and device settings
- Processing timestamps



