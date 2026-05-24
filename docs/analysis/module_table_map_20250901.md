# Module-to-Table Access Pattern Analysis
## Date: September 1, 2025

## Executive Summary

This analysis maps code modules to database tables, categorizing operations by architectural layer and identifying read/write patterns across the greyhound racing prediction system.

## Database Architecture Overview

### Single Database with Logical Separation
- **Single SQLite Database**: `greyhound_racing_data.db`
- **Staging Tables**: `csv_*_staging` prefix for raw ingestion
- **Production Tables**: Core schema without prefixes
- **No Separate Analytics DB**: All data in single database

### Table Namespace Analysis
| Prefix | Purpose | Tables | Examples |
|--------|---------|--------|----------|
| `csv_*_staging` | Raw CSV ingestion | 2 | `csv_dog_history_staging`, `csv_race_metadata_staging` |
| No prefix | Production data | 25+ | `race_metadata`, `dog_race_data`, `enhanced_expert_data` |
| `alembic_*` | Migration tracking | 1 | `alembic_version` |

## Module Access Pattern Analysis

### 1. Ingestion Layer (WRITE-HEAVY)

#### Primary Writers
- **`scripts/ingest_csv_history.py`** 
  - WRITES: `csv_dog_history_staging`, `csv_race_metadata_staging`
  - READS: Staging tables for validation
  - Role: CSV → Staging transformation

- **`scripts/migrate_real_data.py`**
  - WRITES: `race_metadata`, `dog_race_data`, `enhanced_expert_data`
  - READS: `csv_*_staging` tables
  - Role: Staging → Production promotion

- **`ingestion/staging_writer.py`**
  - WRITES: `csv_*_staging` tables
  - Role: Raw CSV parsing and staging

#### Secondary Writers
- **`bulk_csv_ingest.py`**: Batch CSV processing
- **`database_repair_system.py`**: Data quality fixes
- **`scripts/data_integrity_cleanup.py`**: Duplicate resolution

### 2. Feature Building Layer (READ-HEAVY)

#### Core Feature Builders
- **`temporal_feature_builder.py`**
  - READS: `dog_race_data`, `race_metadata`, `enhanced_expert_data`
  - Complex JOINs: `d.race_id = r.race_id AND d.dog_clean_name = e.dog_clean_name`
  - Access Pattern: Historical lookups by dog name

- **`ml_system_v4.py`**
  - READS: All production tables for training data
  - WRITES: Feature caches, model results
  - Role: Primary ML pipeline orchestration

#### Feature Enhancement
- **`tgr_enrichment_service.py`**: External data integration
- **`weather_enhanced_predictor.py`**: Weather feature building

### 3. API/Services Layer (READ-ONLY)

#### API Endpoints
- **`fastapi_app/main.py`**
  - READS: Production tables for predictions
  - WRITES: None (read-only API)
  - Caching: In-memory only

- **`app.py` (Legacy Flask)**
  - READS: All tables for dashboard views
  - WRITES: Minimal caching only

#### Service Components
- **`baseline_stats_manager.py`**: Statistical summaries
- **`monitoring_api.py`**: Health checks and metrics

### 4. Training/Prediction Layer (READ-HEAVY)

#### Model Training
- **`train_model_v4.py`**
  - READS: `dog_race_data`, `race_metadata`, `enhanced_expert_data`
  - WRITES: Model artifacts (files), not database
  - Access Pattern: Bulk data loading for training

- **`scripts/train_optimized_v4.py`**: Production training pipeline

#### Prediction Systems
- **`comprehensive_prediction_pipeline.py`**: Live predictions
- **`probability_calibrator.py`**: Model calibration

### 5. Analytics/Monitoring Layer (READ-HEAVY)

#### Data Analysis
- **`advanced_system_analyzer.py`**: Performance analysis
- **`validate_data_integrity.py`**: Quality checks
- **`temporal_anomaly_investigation.py`**: Data validation

#### System Monitoring
- **`scripts/monitor_system_health.py`**: Health checks
- **`data_monitoring_system.py`**: Data quality monitoring

## RACI Matrix: Module Responsibilities

| Layer | Read Access | Write Access | Cache Write | Validation |
|-------|-------------|--------------|-------------|------------|
| **Ingestion** | ✅ | ✅ Primary | ❌ | ✅ |
| **Feature Building** | ✅ Primary | ❌ | ✅ | ❌ |
| **API/Services** | ✅ | ❌ | ✅ Limited | ❌ |
| **Training** | ✅ Primary | ❌ | ❌ | ❌ |
| **Analytics** | ✅ | ❌ | ❌ | ✅ |

## Architecture Compliance Analysis

### ✅ **STRENGTHS**
1. **Clear Separation**: Staging vs production table separation
2. **Read-Only APIs**: API layer properly read-only
3. **Centralized Writing**: Ingestion layer controls writes
4. **Proper Staging Flow**: CSV → Staging → Production pipeline

### ⚠️ **ARCHITECTURAL CONCERNS**

#### 1. Single Database Limitations
- **Risk**: No separation between operational and analytical workloads
- **Impact**: Query performance conflicts between API and analytics
- **Recommendation**: Consider read replicas for analytics

#### 2. Mixed Access Patterns
- **Issue**: Some scripts blur layer boundaries
- **Example**: `database_repair_system.py` performs both analysis and writes
- **Recommendation**: Separate analysis and repair functions

#### 3. Legacy Components
- **Issue**: `app.py` contains extensive mixed read/write logic
- **Impact**: Maintenance complexity
- **Status**: Being migrated to FastAPI

### 🔴 **VIOLATIONS FOUND**

#### 1. Direct Production Writes Outside Ingestion
```python
# Found in: database_repair_system.py, automated_issue_fixer.py
# Problem: Bypassing staging validation
UPDATE dog_race_data SET finish_position = ? WHERE ...
```

#### 2. Ad-hoc Data Modifications
```python
# Found in: Various repair scripts
# Problem: No centralized data modification tracking
DELETE FROM race_metadata WHERE ...
```

## Query Pattern Analysis

### Most Common Access Patterns
1. **Historical Dog Lookup**: `dog_clean_name` + `race_date` range
2. **Race Join Pattern**: `dog_race_data` ⟨⟩ `race_metadata` ⟨⟩ `enhanced_expert_data`
3. **Temporal Filtering**: Date-based filtering for feature windows

### Index Utilization
- **Effective**: Composite indexes on common JOIN patterns
- **Missing**: Some temporal filtering could benefit from additional indexes

## Recommendations

### Immediate (This Week)
1. **Consolidate Data Modifications**: Channel all writes through ingestion layer
2. **Add Write Audit Trail**: Log all data modifications with source tracking
3. **Separate Analysis from Repair**: Split analysis and modification functions

### Short Term (Next Month)
1. **Read Replica**: Consider analytics read replica to separate workloads
2. **Service Layer**: Formalize service boundaries with clear interfaces
3. **Cache Strategy**: Implement coordinated caching between API and analytics

### Long Term (Next Quarter)
1. **Microservices**: Split into ingestion, feature, and API services
2. **Event Sourcing**: Track all data changes as events
3. **CQRS Pattern**: Separate command and query responsibilities

## Conclusion

The current architecture demonstrates **good separation of concerns** at the logical level, with clear staging-to-production flow and read-only APIs. However, the **single database architecture** creates potential performance conflicts, and some **boundary violations** exist in repair scripts.

The system is **production-ready** for current scale, but architectural evolution will be needed as data volume and concurrent usage grows.
