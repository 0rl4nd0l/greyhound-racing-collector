# Greyhound Racing Database Schema Documentation

**Generated:** January 31, 2025  
**Database:** SQLite (`greyhound_racing_data.db`)  
**Total Tables:** 30 active tables (excluding backup tables)  
**Total Records:** ~35,000+ records across all tables

---

## 📊 Table Summary

| Category | Tables | Purpose |
|----------|--------|---------|
| **Core Data** | 3 | Primary race and dog data |
| **Enhanced/Extra** | 10 | Extended data from multiple sources |
| **Lookup/Reference** | 3 | Mappings and reference data |
| **Analytics** | 2 | Analysis and predictions |
| **Weather** | 5 | Weather data and forecasting |
| **Odds/Betting** | 4 | Live odds and betting analysis |
| **Performance** | 3 | Dog and trainer performance tracking |

---

## 🏗️ Core Data Tables

### 1. race_metadata
**Primary race information table**
- **Rows:** 1,320
- **Primary Key:** `id` (INTEGER, AUTO_INCREMENT)
- **Unique Key:** `race_id` (TEXT)

#### Columns:
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Auto-incrementing ID |
| race_id | TEXT | UNIQUE | Unique race identifier |
| venue | TEXT | | Racing venue code |
| race_number | INTEGER | | Race number at venue |
| race_date | DATE | | Date of race |
| race_name | TEXT | | Name/title of race |
| grade | TEXT | | Race grade (e.g., Grade 5, Grade 4) |
| distance | TEXT | | Race distance in meters |
| track_condition | TEXT | | Track surface condition |
| weather | TEXT | | Weather description |
| temperature | REAL | | Temperature in Celsius |
| humidity | REAL | | Humidity percentage |
| wind_speed | REAL | | Wind speed |
| wind_direction | TEXT | | Wind direction |
| track_record | TEXT | | Track record time |
| prize_money_total | REAL | | Total prize money |
| prize_money_breakdown | TEXT | | Prize distribution |
| race_time | TEXT | | Winning race time |
| field_size | INTEGER | | Number of runners |
| url | TEXT | | Source URL |
| extraction_timestamp | DATETIME | | Data extraction time |
| data_source | TEXT | | Data source identifier |
| winner_name | TEXT | | Name of winning dog |
| winner_odds | REAL | | Winning odds |
| winner_margin | REAL | | Winning margin |
| race_status | TEXT | | Race status |
| data_quality_note | TEXT | | Data quality notes |
| actual_field_size | INTEGER | | Actual number of starters |
| scratched_count | INTEGER | | Number of scratched dogs |
| scratch_rate | REAL | | Percentage of scratches |
| box_analysis | TEXT | | Box draw analysis |
| weather_condition | TEXT | | Detailed weather condition |
| precipitation | REAL | | Precipitation amount |
| pressure | REAL | | Atmospheric pressure |
| visibility | REAL | | Visibility distance |
| weather_location | TEXT | | Weather data location |
| weather_timestamp | DATETIME | | Weather data timestamp |
| weather_adjustment_factor | REAL | | Weather impact factor |
| sportsbet_url | TEXT | | Sportsbet URL |
| venue_slug | TEXT | | URL-friendly venue name |
| start_datetime | DATETIME | | Race start date/time |

#### Indexes:
- `idx_race_metadata_unique` (UNIQUE on race_id)
- `idx_race_metadata_race_date` (on race_date)
- `idx_race_metadata_venue_date` (on venue, race_date)
- `idx_race_metadata_venue` (on venue)
- `idx_race_metadata_extraction` (on extraction_timestamp)
- `idx_race_metadata_race_id` (on race_id)

---

### 2. dog_race_data
**Individual dog performance in races**
- **Rows:** 8,941
- **Foreign Key:** `race_id` → `race_metadata.race_id`

#### Columns (Key):
| Column | Type | Description |
|--------|------|-------------|
| race_id | TEXT | Links to race_metadata |
| dog_name | TEXT | Dog's full name |
| dog_clean_name | TEXT | Normalized dog name |
| box_number | INT | Starting box number |
| finish_position | INTEGER | Final finishing position |
| trainer_name | TEXT | Trainer's name |
| weight | REAL | Dog's racing weight |
| odds_decimal | REAL | Starting odds (decimal) |
| individual_time | TEXT | Individual race time |
| sectional_1st | TEXT | First section time |
| sectional_2nd | TEXT | Second section time |
| margin | TEXT | Winning/losing margin |
| win_probability | REAL | Calculated win probability |
| place_probability | REAL | Calculated place probability |

#### Indexes:
- `idx_dog_race_unique` (UNIQUE on race_id, dog_clean_name, box_number)
- `idx_dog_race_data_race_id` (on race_id)
- `idx_dog_race_data_dog_name` (on dog_name)
- `idx_dog_name` (on dog_clean_name)
- `idx_dog_race_data_finish_position` (on finish_position)

---

### 3. dogs
**Master dog registry**
- **Rows:** 11,920
- **Primary Key:** `dog_id` (INTEGER, AUTO_INCREMENT)

#### Columns:
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| dog_id | INTEGER | PRIMARY KEY | Auto-incrementing dog ID |
| dog_name | TEXT | UNIQUE, NOT NULL | Dog's name |
| total_races | INTEGER | DEFAULT 0 | Total races run |
| total_wins | INTEGER | DEFAULT 0 | Total wins |
| total_places | INTEGER | DEFAULT 0 | Total placings (1st-3rd) |
| best_time | REAL | | Best recorded time |
| average_position | REAL | | Average finishing position |
| last_race_date | TEXT | | Date of last race |
| weight | DECIMAL(5,2) | | Typical racing weight |
| age | INTEGER | | Age in months |
| color | TEXT | | Dog's color |
| owner | TEXT | | Owner's name |
| trainer | TEXT | | Trainer's name |
| sex | TEXT | | Dog's sex (M/F) |

---

## 🔗 Enhanced Data Tables

### 4. enhanced_expert_data
**Expert form analysis data**
- **Rows:** 3,113
- **Unique Key:** `race_id, dog_clean_name`

#### Key Features:
- PIR (Performance Index Rating) ratings
- Sectional times and margins
- Expert form analysis
- Starting prices and positions

---

### 5. dogs_ft_extra
**FastTrack extended dog information**
- **Rows:** 0 (structure ready)
- **Foreign Key:** `dog_id` → `dogs.id`

#### Extended Information:
- Breeding information (sire, dam)
- Career statistics
- Winning box analysis
- Whelping dates and ages

---

### 6. races_ft_extra
**FastTrack extended race information**
- **Rows:** 0 (structure ready)
- **Foreign Key:** `race_id` → `races.id`

#### Extended Information:
- Track ratings
- Split times for winners
- Video URLs
- Stewards' reports

---

### 7. expert_form_analysis
**Expert form analysis from PDFs**
- **Rows:** 0 (structure ready)
- **Foreign Key:** `race_id` → `races.id`

#### Analysis Features:
- Expert selections
- Confidence ratings
- Key insights extraction
- Processing status tracking

---

### 8. gr_* Tables (The Greyhound Recorder)
**Extended data from The Greyhound Recorder source**

#### gr_race_details
- Race-specific details
- Prize money and conditions
- Track and weather conditions

#### gr_dog_entries
- Dog entry information
- Form guides and recent form
- Trainer and owner details

#### gr_dog_form
- Historical form data
- Split times and margins
- Race-specific performance

---

## 📈 Analytics & Predictions

### 9. race_analytics
**Race analysis and predictions**
- **Rows:** 0 (structure ready)
- **Foreign Key:** `race_id` → `race_metadata.race_id`

#### Analysis Types:
- Predicted winners
- Confidence scores
- Model versions
- Analysis timestamps

---

### 10. gpt_analysis
**GPT-powered race analysis**
- **Rows:** 0 (structure ready)
- **Foreign Key:** `race_id` → `race_metadata.race_id`

#### AI Analysis:
- Token usage tracking
- Model identification
- Confidence scoring
- Multiple analysis types

---

## 🌤️ Weather Data System

### 11. weather_data_v2
**Primary weather data (Open Meteo)**
- **Rows:** 349
- **Unique Key:** `venue_code, race_date, race_time`

#### Weather Metrics:
- Temperature, humidity, pressure
- Wind speed and direction
- Precipitation and visibility
- Weather codes and conditions

---

### 12. weather_data
**Legacy weather data (Mock API)**
- **Rows:** 3
- **Purpose:** Fallback weather source

---

### 13. weather_impact_analysis
**Weather impact on race performance**
- **Rows:** 0 (structure ready)
- **Purpose:** Analyze weather effects on outcomes

---

### 14. weather_forecast_cache
**Cached weather forecasts**
- **Rows:** 0 (structure ready)
- **Purpose:** Cache forecast data with expiration

---

## 💰 Odds & Betting System

### 15. live_odds
**Real-time odds data**
- **Rows:** 307
- **Source:** Sportsbet integration

#### Odds Tracking:
- Decimal and fractional odds
- Market types (win, place)
- Timestamp tracking
- Current status flags

---

### 16. odds_history
**Historical odds movements**
- **Rows:** 0 (structure ready)
- **Purpose:** Track odds changes over time

---

### 17. value_bets
**Value betting opportunities**
- **Rows:** 0 (structure ready)
- **Purpose:** Identify value betting situations

---

### 18. predictions
**ML model predictions**
- **Rows:** 0 (structure ready)
- **Purpose:** Store model predictions

---

## 🏃 Performance Tracking

### 19. dog_performances
**Simplified dog performance tracking**
- **Rows:** 8,225
- **Purpose:** Streamlined performance data

---

### 20. trainers
**Trainer performance statistics**
- **Rows:** 0 (structure ready)
- **Purpose:** Track trainer success rates

---

### 21. trainer_performance
**Detailed trainer analysis**
- **Rows:** 0 (structure ready)
- **Purpose:** Advanced trainer statistics

---

## 🗂️ Lookup & Reference Tables

### 22. venue_mappings
**Venue code mappings**
- **Rows:** 38
- **Purpose:** Map venue codes to official names

#### Venue Resolution:
- Official names and codes
- Location information
- Active status tracking

---

### 23. comprehensive_dog_profiles
**Complete dog profiles**
- **Rows:** 0 (structure ready)
- **Purpose:** Comprehensive dog statistics

---

### 24. detailed_race_history
**Extended race history**
- **Rows:** 0 (structure ready)
- **Purpose:** Detailed historical performance

---

### 25. track_conditions
**Track condition data**
- **Rows:** 0 (structure ready)
- **Purpose:** Track surface and condition tracking

---

### 26. alembic_version
**Database migration version**
- **Rows:** 1
- **Purpose:** Track database schema versions

---

## 🔗 Relationships & Foreign Keys

### Primary Relationships:
1. **race_metadata** ← **dog_race_data** (via race_id)
2. **dogs** ← **dogs_ft_extra** (via dog_id)
3. **race_metadata** ← **races_ft_extra** (via race_id)
4. **race_metadata** ← **gr_race_details** (via race_id)
5. **gr_dog_entries** ← **gr_dog_form** (via dog_entry_id)
6. **comprehensive_dog_profiles** ← **detailed_race_history** (via dog_name)

### Data Flow:
```
race_metadata (1,320 races)
├── dog_race_data (8,941 performances)
├── enhanced_expert_data (3,113 expert records)
├── live_odds (307 odds records)
├── weather_data_v2 (349 weather records)
└── Various analysis tables

dogs (11,920 dogs)
├── dog_performances (8,225 simplified performances)
└── Extended profile tables
```

---

## 📊 Data Quality & Integrity

### Active Data:
- **race_metadata:** 1,320 races with winners and complete data
- **dog_race_data:** 8,941 individual performances
- **dogs:** 11,920 unique dogs in system
- **enhanced_expert_data:** 3,113 expert form records

### Backup Tables:
Multiple backup tables preserve historical data during migrations and updates:
- `dog_race_data_backup`
- `dog_race_data_backup_box_number_fix`
- `race_metadata_backup_dedup_race_metadata`
- `enhanced_expert_data_backup_dog_day_fix`

### Data Sources:
1. **FastTrack** - Official racing data
2. **The Greyhound Recorder** - Form guides and detailed information
3. **Sportsbet** - Live odds and betting data
4. **Open Meteo** - Weather data
5. **Expert Forms** - Professional form analysis

---

## 🎯 Key Performance Indicators

### Database Health:
- **Total Records:** ~35,000+ across all tables
- **Referential Integrity:** All foreign keys properly configured
- **Indexing:** Comprehensive indexing for performance
- **Data Coverage:** Multi-source data integration

### Recent Activity:
- Active race collection and processing
- Real-time odds integration
- Weather data synchronization
- Form analysis automation

---

## 🔧 Schema Evolution

### Migration System:
- **Alembic** version control
- Backup strategies for all schema changes  
- Incremental updates with rollback capability

### Recent Changes:
- Enhanced weather integration
- Improved foreign key relationships
- Performance optimization indexes
- Multi-source data consolidation

---

*This documentation was auto-generated from the live database schema on January 31, 2025. For the most current schema information, regenerate this documentation or examine the live database directly.*
