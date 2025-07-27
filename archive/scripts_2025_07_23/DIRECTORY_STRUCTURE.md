# Clean Directory Structure

## 📁 Root Directory - Only Essential Files

```
greyhound_racing_collector/
├── run.py                          # ⭐ Main entry point
├── form_guide_csv_scraper.py       # ⭐ Form guide CSV downloader
├── advanced_ai_analysis.py         # ⭐ Advanced AI analysis
├── advanced_scraper.py             # ⭐ Advanced web scraping
├── README.md                       # Quick start guide
├── requirements.txt                # Basic dependencies
├── requirements_advanced.txt       # Advanced dependencies
└── DIRECTORY_STRUCTURE.md          # This file
```

## 📂 Organized Subdirectories

### `/data/` - All data files
- `confidence_weighted_insights.json`
- `data_confidence_report.json`
- `form_guide_progress.json`
- `greyhound_analytics_pipeline_summary.json`
- `race_confidence_scores.csv`
- `record_confidence_scores.csv`
- `ultimate_insights.json`

### `/databases/` - Database files
- `comprehensive_greyhound_data.db`

### `/docs/` - Documentation
- `PIPELINE_SUMMARY.md`
- `SYSTEM_ARCHITECTURE.md`
- `SYSTEM_FINAL.md`
- `SYSTEM_STATUS.md`
- `ULTIMATE_ANALYSIS_CAPABILITIES.md`

### `/form_guides/` - Form guide data
- `processed/` - Processed form guides (935+ CSV files)
- `unprocessed/` - New form guides waiting to be processed
- Various JSON and CSV files

### `/prediction_agent/` - Prediction system
- `predict.py` - Main prediction interface
- `dynamic_prediction.py` - Dynamic prediction logic
- `insights_loader.py` - Insights loading
- `venv/` - Virtual environment

### `/unprocessed/` - Raw form guides
- CSV files waiting to be processed

### `/outdated_scripts/` - Archived scripts
- All legacy scripts moved here for reference

## 🎯 Key Benefits

1. **Clean Root**: Only 4 essential Python scripts + config files
2. **Organized Data**: All data files in dedicated folders
3. **Clear Purpose**: Each folder has a specific function
4. **Easy Navigation**: Find everything quickly
5. **Maintainable**: Easy to add new features without clutter

## 🚀 Usage

The system is now ultra-clean and organized:

```bash
# Main operations (all from root directory)
python3 run.py help            # Show help
python3 run.py collect         # Download form guide CSVs
python3 run.py advanced        # Run advanced AI analysis
python3 run.py predict <file>  # Make predictions
```

All data, databases, and documentation are properly organized in their respective folders!
