# Outdated Scripts Archive

This directory contains scripts that have been superseded by newer, more advanced versions in the main greyhound racing collector system.

## Moved on: July 11, 2025

### Prediction Agent Scripts (Superseded)

These scripts were replaced by the new dynamic prediction system:

#### **Moved to `prediction_agent/`:**
- `run_race_prediction.py` - Old race prediction runner
- `prediction_assistant.py` - Legacy prediction assistant
- `simple_prediction.py` - Basic prediction script (superseded by dynamic_prediction.py)
- `smart_prediction.py` - Smart prediction with dataset copying (superseded by predict.py)
- `standalone_prediction.py` - Standalone prediction script (superseded)
- `ultimate_prediction.py` - Ultimate prediction script (superseded)
- `ai_enhanced_prediction.py` - AI-enhanced prediction (superseded)
- `predict_enhanced.py` - Enhanced prediction script (redundant with predict.py)

**Replaced by:**
- `predict.py` - Main prediction entry point with dynamic insights loading and dataset copying
- `dynamic_prediction.py` - Dynamic prediction system that automatically uses the most recent analysis
- `insights_loader.py` - Dynamic insights loading system

### Data Collection Scripts (Superseded)

#### **Moved to root:**
- `enhanced_odds_collector.py` - Enhanced odds collection system (superseded by form_guide_scraper_2025.py)
- `greyhound_results_scraper_navigator.py` - Results scraper with navigation (superseded)
- `integrated_race_collector.py` - Integrated race data collector (superseded)
- `test_race_download.py` - Test script for downloads (temporary)

**Replaced by:**
- `form_guide_scraper_2025.py` - Current active scraper
- `comprehensive_form_guide_processor.py` - Form processing system

## Current Active System

### **Core Pipeline (Active):**
- `form_guide_scraper_2025.py` - Main data scraper
- `comprehensive_form_guide_processor.py` - Form guide processing
- `data_confidence_system.py` - Data confidence scoring
- `confidence_weighted_analysis.py` - Main analysis with confidence weighting
- `ultimate_comprehensive_analysis.py` - Comprehensive analysis system
- `pipeline_summary_report.py` - Pipeline reporting

### **Prediction System (Active):**
- `prediction_agent/predict.py` - Main prediction entry point
- `prediction_agent/dynamic_prediction.py` - Dynamic prediction system
- `prediction_agent/insights_loader.py` - Automatic insights loading

## Why These Were Moved

1. **Redundancy**: Multiple scripts doing similar functions
2. **Superseded**: Newer versions with better features
3. **Maintenance**: Reduces confusion and maintenance burden
4. **Evolution**: System has evolved to use dynamic insights loading

## Key Improvements in New System

- **Dynamic Insights Loading**: Automatically uses the most recent analysis data
- **Confidence Weighting**: Incorporates data quality into predictions
- **Multi-track Support**: Handles all 44 venues automatically
- **Future-proof**: Adapts to new analysis formats automatically
- **Simplified Interface**: Single entry point for all predictions

## Recovery

If any of these scripts are needed again, they can be moved back from this directory. They are preserved exactly as they were when moved.

---

*Archive created: July 11, 2025*
*System cleaned up to focus on the active dynamic prediction pipeline*
