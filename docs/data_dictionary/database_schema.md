# Database Schema

This document provides a comprehensive data dictionary for the Greyhound Racing Predictor database schema.

## Overview

The database stores race data, greyhound information, weather conditions, and model performance metrics. The schema is designed to support both historical analysis and real-time predictions.

## Core Tables

### `races`
Stores information about individual races.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `id` | INTEGER | Primary key, unique race identifier | NOT NULL, PRIMARY KEY |
| `track_id` | INTEGER | Foreign key to tracks table | NOT NULL |
| `race_number` | INTEGER | Race number for the day | NOT NULL |
| `race_date` | DATE | Date of the race | NOT NULL |
| `race_time` | TIME | Scheduled start time | |
| `distance` | INTEGER | Race distance in meters | |
| `grade` | VARCHAR(10) | Race grade (A1, A2, etc.) | |
| `track_condition` | VARCHAR(20) | Track surface condition | |
| `weather_id` | INTEGER | Foreign key to weather table | |

### `greyhounds`
Stores greyhound information and statistics.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `id` | INTEGER | Primary key, unique greyhound identifier | NOT NULL, PRIMARY KEY |
| `name` | VARCHAR(100) | Greyhound name | NOT NULL |
| `trainer` | VARCHAR(100) | Trainer name | |
| `age` | INTEGER | Age in months | |
| `weight` | DECIMAL(5,2) | Weight in kilograms | |
| `sex` | CHAR(1) | Gender (M/F) | |
| `color` | VARCHAR(20) | Coat color | |
| `sire` | VARCHAR(100) | Father's name | |
| `dam` | VARCHAR(100) | Mother's name | |

### `race_entries`
Links greyhounds to specific races with their performance data.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `id` | INTEGER | Primary key | NOT NULL, PRIMARY KEY |
| `race_id` | INTEGER | Foreign key to races table | NOT NULL |
| `greyhound_id` | INTEGER | Foreign key to greyhounds table | NOT NULL |
| `trap_number` | INTEGER | Starting trap (1-8) | NOT NULL |
| `finishing_position` | INTEGER | Final position (1st, 2nd, etc.) | |
| `run_time` | DECIMAL(6,3) | Race time in seconds | |
| `sectional_times` | JSON | Split times at various points | |
| `starting_price` | DECIMAL(8,2) | Betting odds | |
| `weight` | DECIMAL(5,2) | Race weight | |
| `comment` | TEXT | Race comments | |

### `weather`
Weather conditions for race days.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `id` | INTEGER | Primary key | NOT NULL, PRIMARY KEY |
| `date` | DATE | Weather date | NOT NULL |
| `track_id` | INTEGER | Foreign key to tracks table | NOT NULL |
| `temperature` | DECIMAL(4,1) | Temperature in Celsius | |
| `humidity` | INTEGER | Humidity percentage | |
| `wind_speed` | DECIMAL(4,1) | Wind speed in km/h | |
| `wind_direction` | VARCHAR(10) | Wind direction | |
| `precipitation` | DECIMAL(5,2) | Rainfall in mm | |
| `visibility` | INTEGER | Visibility in km | |

### `tracks`
Information about racing tracks.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `id` | INTEGER | Primary key | NOT NULL, PRIMARY KEY |
| `name` | VARCHAR(100) | Track name | NOT NULL |
| `location` | VARCHAR(100) | City/State | |
| `surface` | VARCHAR(20) | Track surface type | |
| `circumference` | INTEGER | Track length in meters | |
| `timezone` | VARCHAR(50) | Local timezone | |

### `model_performance`
Tracks model accuracy and performance metrics.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `id` | INTEGER | Primary key | NOT NULL, PRIMARY KEY |
| `model_name` | VARCHAR(100) | Model identifier | NOT NULL |
| `evaluation_date` | TIMESTAMP | When metrics were calculated | NOT NULL |
| `accuracy` | DECIMAL(5,4) | Overall accuracy | |
| `precision` | DECIMAL(5,4) | Precision score | |
| `recall` | DECIMAL(5,4) | Recall score | |
| `f1_score` | DECIMAL(5,4) | F1 score | |
| `auc_roc` | DECIMAL(5,4) | Area under ROC curve | |
| `sample_size` | INTEGER | Number of predictions evaluated | |

## Relationships

- `races` → `tracks` (many-to-one)
- `races` → `weather` (many-to-one)
- `race_entries` → `races` (many-to-one)
- `race_entries` → `greyhounds` (many-to-one)
- `weather` → `tracks` (many-to-one)

## Indexes

### Primary Indexes
- All tables have primary key indexes on their `id` columns

### Secondary Indexes
- `races.race_date` - for date range queries
- `races.track_id` - for track-specific queries
- `race_entries.race_id` - for race lookup
- `race_entries.greyhound_id` - for greyhound performance history
- `greyhounds.name` - for name searches
- `weather.date, weather.track_id` - composite index for weather lookups

## Data Types and Constraints

### Validation Rules
- Race distances typically range from 300m to 800m
- Trap numbers are between 1 and 8
- Finishing positions are between 1 and 8 (or NULL for non-finishers)
- Weights are typically between 25kg and 40kg
- Times are positive decimal values

### Business Rules
- Each race must have a unique combination of track, date, and race number
- Greyhound names should be unique within the system
- Weather data should be recorded daily for each active track
- Model performance metrics should be updated regularly

## Data Retention

- Race data: Retained indefinitely for historical analysis
- Weather data: Retained for 5 years
- Model performance: Retained for 2 years
- Temporary prediction data: Purged after 30 days
