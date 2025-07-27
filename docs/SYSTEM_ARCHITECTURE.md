# Greyhound Racing Collector - System Architecture

## 🎯 Overview

This document describes the current active system architecture after the cleanup performed on July 11, 2025.

## 📁 Directory Structure

```
greyhound_racing_collector/
├── 📊 DATA COLLECTION & PROCESSING
│   ├── form_guide_scraper_2025.py          # Main data scraper
│   ├── comprehensive_form_guide_processor.py # Form guide processing
│   ├── download_2025_forms.py              # Form download utility
│   └── comprehensive_greyhound_data.db     # Main database
│
├── 🔬 ANALYSIS PIPELINE
│   ├── data_confidence_system.py           # Data confidence scoring
│   ├── confidence_weighted_analysis.py     # Main analysis with confidence weighting
│   ├── ultimate_comprehensive_analysis.py  # Comprehensive analysis system
│   └── pipeline_summary_report.py          # Pipeline reporting
│
├── 🎯 PREDICTION SYSTEM
│   └── prediction_agent/
│       ├── predict.py                      # Main prediction entry point
│       ├── dynamic_prediction.py           # Dynamic prediction system
│       ├── insights_loader.py              # Automatic insights loading
│       └── form_guides/                    # Race form guides
│
├── 📋 INSIGHTS & REPORTS
│   ├── confidence_weighted_insights.json   # Latest analysis results
│   ├── ultimate_insights.json             # Comprehensive analysis results
│   └── PIPELINE_SUMMARY.md                # Pipeline documentation
│
└── 🗂️ ARCHIVE
    └── outdated_scripts/                   # Superseded scripts
        ├── prediction_agent/               # Old prediction scripts
        └── README.md                       # Archive documentation
```

## 🔄 Active System Workflow

### 1. **Data Collection**
```bash
# Scrape new race forms
python form_guide_scraper_2025.py

# Process form guides
python comprehensive_form_guide_processor.py
```

### 2. **Analysis Pipeline**
```bash
# Generate confidence-weighted analysis
python confidence_weighted_analysis.py

# Generate comprehensive analysis
python ultimate_comprehensive_analysis.py
```

### 3. **Prediction System**
```bash
# Run predictions (uses most recent analysis automatically)
python prediction_agent/predict.py "Race 2 - HEA - 11 July 2025.csv"
```

## 🎯 Key Features

### **Dynamic Insights Loading**
- Automatically detects and uses the most recent analysis file
- Priority order: confidence_weighted → ultimate → ultra → advanced
- No manual configuration required

### **Confidence-Weighted Analysis**
- Novel approach to handle data quality variations
- Weighted statistics based on data completeness
- Uncertainty quantification for all results

### **Multi-Track Support**
- Comprehensive database of 44 venues
- Track-specific bias analysis
- Automatic venue detection from race data

### **Future-Proof Architecture**
- Adapts to new analysis formats automatically
- Backward compatible with existing data
- Scalable for additional features

## 📊 Current Data Status

- **Total Records**: 6,827 race entries
- **Total Races**: 1,144 races analyzed
- **Venues**: 44 tracks with bias data
- **Analysis Type**: Confidence-weighted (most advanced)
- **Confidence Score**: 92.6% average confidence

## 🚀 Usage Examples

### **Basic Prediction**
```bash
cd prediction_agent
python predict.py "Race 2 - HEA - 11 July 2025.csv"
```

### **Check Available Race Files**
```bash
cd prediction_agent
python predict.py
```

### **View Analysis Status**
```bash
cd prediction_agent
python insights_loader.py
```

## 🔧 System Benefits

1. **Simplified Architecture**: Reduced from 19 scripts to 7 core scripts
2. **Automated Insights**: No manual file management required
3. **Confidence Weighting**: Data quality integrated into predictions
4. **Track Adaptability**: Works with all Australian greyhound tracks
5. **Future-Ready**: Automatically adapts to system improvements

## 📈 What Was Cleaned Up

### **Moved to Archive:**
- **8 outdated prediction scripts** (superseded by dynamic system)
- **4 legacy data collection scripts** (superseded by current scraper)
- **1 test script** (no longer needed)

### **Key Consolidations:**
- Multiple prediction scripts → Single `predict.py` with dynamic loading
- Manual file management → Automatic insights detection
- Static analysis → Confidence-weighted analysis
- Track-specific scripts → Universal multi-track support

## 🎯 Next Steps

The system is now optimized for:
- **Continuous data collection** using the active scraper
- **Automatic analysis updates** with the confidence-weighted system
- **Dynamic predictions** that always use the latest insights
- **Easy expansion** as new features are added

## 📞 Support

All scripts include comprehensive help text and error handling. Use the `--help` flag or run without arguments to see usage information.

---

*System Architecture - Updated July 11, 2025*
*Clean, efficient, and future-ready greyhound racing analytics pipeline*
