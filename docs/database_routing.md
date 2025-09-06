# Database Routing System

## Overview

The Greyhound Racing Collector now uses a dual-database architecture to separate staging (write) and analytics (read) operations. This improves performance, data integrity, and allows for different optimization strategies for each workload type.

## Architecture

The system routes database operations through a centralized `db_utils.py` module:

- **Staging Database**: Receives writes from data ingestion, model training, and registration
- **Analytics Database**: Provides optimized read access for evaluation, prediction, and analysis

## Environment Variables

### Primary Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `STAGING_DB_PATH` | Database for write operations | `./greyhound_racing_data_stage.db` |
| `ANALYTICS_DB_PATH` | Database for read operations | `./greyhound_racing_data_analytics.db` |
| `GREYHOUND_DB_PATH` | Fallback for both read/write when specific paths not set | `./greyhound_racing_data.db` |

### Legacy Support

| Variable | Purpose | Notes |
|----------|---------|-------|
| `DATABASE_PATH` | Legacy database path | Still supported for backward compatibility |
| `DATABASE_URL` | SQLAlchemy-style URL | For tools expecting URL format |

## Usage

### In Scripts

Scripts automatically use the appropriate database based on their operation type:

```python
# Write operations (ingestion, training)
from scripts.db_utils import open_sqlite_writable
conn = open_sqlite_writable(db_path)  # Uses STAGING_DB_PATH

# Read operations (evaluation, analysis)  
from scripts.db_utils import open_sqlite_readonly
conn = open_sqlite_readonly(db_path)  # Uses ANALYTICS_DB_PATH
```

### Script Categories

#### Write Operations (→ Staging DB)
- Data ingestion scripts (`ingest_*.py`)
- Model training scripts (`train_*.py`) 
- Model registration (`register_*.py`)
- Schema patching (`verify_and_patch_schema.py`)
- TGR backfilling (`tgr_backfill_*.py`)

#### Read Operations (→ Analytics DB)
- Model evaluation (`evaluate_*.py`)
- Race prediction (`predict_*.py`)
- Analysis and reporting scripts
- Development/testing scripts (`dev/check_*.py`)

## Database Synchronization

In production deployments, you'll need to regularly synchronize data from the staging database to the analytics database. This can be done via:

1. **Full copy**: Periodic complete database replacement
2. **Incremental sync**: Sync only new/updated records
3. **ETL pipeline**: Transform and optimize data during sync

Example sync command:
```bash
# Simple approach - copy staging to analytics
cp greyhound_racing_data_stage.db greyhound_racing_data_analytics.db

# Or use SQLite backup command for online copy
sqlite3 greyhound_racing_data_stage.db ".backup greyhound_racing_data_analytics.db"
```

## Benefits

### Performance
- **Staging DB**: Optimized for write operations, faster ingestion
- **Analytics DB**: Optimized for complex queries, better read performance

### Data Integrity  
- Staging operations don't interfere with ongoing analysis
- Analytics database remains stable during data ingestion

### Flexibility
- Different backup strategies for each database type
- Can use different storage locations/types for each workload
- Easier to scale read vs write operations independently

## Development Setup

### Single Database Mode
For development, you can point both paths to the same file:
```bash
export STAGING_DB_PATH="./greyhound_racing_data.db"
export ANALYTICS_DB_PATH="./greyhound_racing_data.db"
```

### Dual Database Mode
For testing the routing system:
```bash
export STAGING_DB_PATH="./data_stage.db"
export ANALYTICS_DB_PATH="./data_analytics.db" 
```

### Legacy Mode  
Use the fallback for simple setups:
```bash
export GREYHOUND_DB_PATH="./greyhound_racing_data.db"
# Both reads and writes will use this database
```

## Migration Guide

### From Single Database
1. Set `GREYHOUND_DB_PATH` to your current database
2. Scripts will continue to work normally using the fallback
3. Optionally migrate to dual-database setup when ready

### To Dual Database
1. Copy your current database to both staging and analytics paths:
   ```bash
   cp current.db greyhound_racing_data_stage.db
   cp current.db greyhound_racing_data_analytics.db
   ```

2. Set environment variables:
   ```bash
   export STAGING_DB_PATH="./greyhound_racing_data_stage.db"
   export ANALYTICS_DB_PATH="./greyhound_racing_data_analytics.db"
   ```

3. Set up regular synchronization from staging to analytics

## Troubleshooting

### Database Not Found
- Check that the specified database files exist
- Verify environment variables are set correctly
- Check file permissions for read/write access

### Performance Issues
- Ensure analytics database is optimized (run `ANALYZE` and `VACUUM`)
- Consider using WAL mode for better concurrent access
- Monitor database sizes and implement archiving if needed

### Data Sync Issues
- Implement monitoring to detect sync lag
- Use transaction-based sync to maintain consistency
- Consider using database triggers or CDC for real-time sync
