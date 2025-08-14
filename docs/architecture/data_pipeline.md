# Data Pipeline

The data pipeline in the Greyhound Racing Predictor system consists of multiple stages:

1. **Ingestion**
   - Data is ingested from multiple sources, including CSV files, web scraping, and APIs.
   - Automated scraping pulls race data, odds, and weather information.

2. **Processing**
   - Data is processed to clean, normalize, and integrate it.
   - Advanced processors enhance data with weather and track conditions.

3. **Storage**
   - Data is stored in a comprehensive SQLite database.
   - The database schema includes tables for race metadata, dog performance, and predictions.

4. **Enhancement**
   - Data is enhanced with additional computed features, such as race predictions and trends.
   - The `EnhancedComprehensiveProcessor` class handles AI-powered analysis.

5. **Validation**
   - Data integrity checks and schema validation ensure quality and consistency.

This entire pipeline is designed to facilitate accurate and reliable greyhound race predictions.
