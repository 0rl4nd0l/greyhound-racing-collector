# Greyhound Racing Data File Naming Standards

**Version:** 1.0  
**Created:** 2025-07-26

## Overview
Standardized naming conventions to prevent duplicates and improve organization

## File Naming Standards

### Race Data
- **Pattern:** `Race_{race_num}_{venue}_{date}.csv`
- **Example:** `Race_01_AP_K_2025-07-26.csv`
- **Description:** Race data files with standardized format

### Form Guides
- **Pattern:** `FormGuide_{venue}_{date}_{race_num}.csv`
- **Example:** `FormGuide_BAL_2025-07-26_01.csv`
- **Description:** Form guide files with venue and date

### Enhanced Analysis
- **Pattern:** `Analysis_{type}_{venue}_{date}_{timestamp}.json`
- **Example:** `Analysis_ML_AP_K_2025-07-26_143022.json`
- **Description:** Enhanced analysis with type and timestamp

### Upcoming Races
- **Pattern:** `Upcoming_{venue}_{date}_{race_num}.csv`
- **Example:** `Upcoming_GEE_2025-07-27_05.csv`
- **Description:** Upcoming race predictions

## Naming Rules

### General
- Use underscores (_) as separators, not spaces or hyphens
- Use YYYY-MM-DD format for dates
- Use uppercase for venue codes (AP_K, BAL, GEE, etc.)
- Use zero-padded numbers for race numbers (01, 02, etc.)
- Include timestamp for files that might be generated multiple times
- No special characters except underscores and hyphens in dates

### Venues
- Use standard venue codes: AP_K, APWE, BAL, BEN, CANN, CASO
- DAPT, GEE, GOSF, GRDN, HEA, HOR, MAND, MOUNT, MURR
- NOR, QOT, RICH, SAL, SAN, TRA, WAR, W_PK

### Dates
- Always use YYYY-MM-DD format
- Future dates for upcoming races, past dates for historical data

### File Types
- Use .csv for tabular race data
- Use .json for analysis results and metadata
- Use .txt for logs and configuration files

## Examples

### ✅ Good Examples
- `Race_01_AP_K_2025-07-26.csv`
- `FormGuide_BAL_2025-07-26_01.csv`
- `Analysis_ML_GEE_2025-07-26_143022.json`
- `Upcoming_RICH_2025-07-27_08.csv`

### ❌ Bad Examples
- `Race 1 - AP_K - 26 July 2025.csv`
- `race1-apk-26-7-25.csv`
- `Race_1_AP_K_26_July_2025_1.csv`
- `form guide bal 26-07-2025.csv`

## Validation
Run `python file_naming_validator.py` to validate existing files against these standards.

## Enforcement
Automated validation runs on data collection
