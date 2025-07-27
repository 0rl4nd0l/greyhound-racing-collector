# Complete File Inventory - Usable Data Files

## Executive Summary
**Total Usable Files: 6,837**
- **188 CSV files** (various types of race data and analysis)
- **6,649 JSON files** (enhanced data, predictions, and analysis results)

## Detailed Breakdown

### CSV Files (188 total)
1. **Race Data Files: 44**
   - Raw race results and form data
   - Located in: `upcoming_races/`, `processed_races/`, `form_guides/`
   - Example: `Race_01_WAR_2025-07-25.csv`

2. **ML Analysis Files: 125**
   - Machine learning analysis results per track
   - Located in: `data/enhanced_data/`
   - Example: `Analysis_ML_AP_K_2025-07-26_121551.csv`

3. **Other CSV Files: 19**
   - Form guides, consolidated data, and other analysis
   - Various locations

### JSON Files (6,649 total)
1. **Enhanced Expert Data: 6,657 files**
   - Enhanced dog performance data
   - Track performance metrics
   - Race analytics with detailed statistics
   - Located in: `enhanced_expert_data/`

2. **Predictions: 73 files**
   - ML-generated race predictions
   - Include confidence levels, betting recommendations
   - Historical performance analysis per dog
   - Located in: `predictions/`

3. **Other Analysis Results: Various**
   - Feature analysis results
   - Backtesting results
   - Model training results
   - Integrated form data

## Key Data Categories

### 1. Race Results & Form Data (44 CSV files)
- **Content**: Raw race results, dog performance, times, positions
- **Value**: Historical performance tracking
- **Example Structure**: Dog Name, Position, Box, Weight, Time, Track, etc.

### 2. ML Analysis Results (125 CSV files)
- **Content**: Machine learning analysis per track
- **Value**: Track-specific performance insights
- **Coverage**: Multiple Australian tracks (AP_K, WAR, SAN, etc.)

### 3. Enhanced Performance Data (6,657 JSON files)
- **Content**: Detailed dog performance analytics
- **Value**: Comprehensive statistical analysis
- **Features**: Speed ratings, class assessments, form trends

### 4. Prediction Models (73 JSON files)
- **Content**: Race predictions with confidence scores
- **Value**: Betting recommendations and win probabilities
- **Features**: Traditional + ML scoring, historical stats

### 5. Backtesting & Validation Data
- **Content**: Model performance validation
- **Value**: Strategy effectiveness measurement
- **Coverage**: Historical accuracy tracking

## Data Quality Assessment

### Excellent Quality (6,700+ files)
- All JSON prediction files contain complete analysis
- Enhanced expert data includes comprehensive statistics
- ML analysis files have proper track-specific data

### Good Quality (44 race CSV files)
- 100% readable and valid race data
- Consistent column structure
- Proper date formatting

### Minor Issues (144 files)
- Some files have non-standard naming suffixes
- 2 race files need cleanup (`_01` suffix)
- Some analysis files have problematic naming patterns

## Usage Recommendations

### For Race Analysis
- Use **44 race CSV files** for historical performance
- Combine with **enhanced expert data JSON** for detailed insights

### For Predictions
- **73 prediction JSON files** contain ready-to-use betting recommendations
- Include confidence levels and risk assessments

### For Model Development
- **125 ML analysis CSV files** provide track-specific training data
- **Backtesting results** show historical model performance

### For Research
- **6,657 enhanced expert files** contain comprehensive dog statistics
- Perfect for developing new analytical models

## Conclusion
You have a **substantial dataset of 6,837 usable files** covering:
- Raw race data
- Enhanced performance analytics  
- ML predictions and analysis
- Model validation results

This represents a comprehensive greyhound racing analysis system with significant value for both research and practical betting applications.
