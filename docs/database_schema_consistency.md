# Database Schema & Migration Consistency Testing

This document describes the comprehensive database schema and migration consistency testing system implemented for the Greyhound Racing Predictor application.

## Overview

The system implements all the requirements specified in Step 4:
- ✅ Run `alembic revision --autogenerate --compare-type` in CI; assert generated diff is **empty**
- ✅ Automated check that every foreign-key pair has existing indexes
- ✅ Integrity queries: orphan records, NULLs in non-nullable columns, enum mismatches
- ✅ Daily cron CI job dumps prod schema → compares hash to repo SQL; alerts on drift

## Components

### 1. Core Testing Framework (`tests/test_database_schema_consistency.py`)

The main testing framework provides comprehensive schema validation:

#### Features:
- **Alembic Schema Consistency**: Verifies that `alembic revision --autogenerate` produces no changes
- **Foreign Key Index Validation**: Ensures all foreign keys have corresponding indexes for performance
- **Data Integrity Checks**: Validates orphan records, NULL violations, and constraint violations
- **Schema Hash Generation**: Creates deterministic hashes for drift detection

#### Usage:
```bash
# Run all schema consistency tests
make schema-tests

# Run individual tests
pytest tests/test_database_schema_consistency.py::test_alembic_schema_consistency -v
pytest tests/test_database_schema_consistency.py::test_foreign_key_indexes -v
pytest tests/test_database_schema_consistency.py::test_data_integrity -v

# Command-line interface
python tests/test_database_schema_consistency.py --database-url="sqlite:///greyhound_racing_data.db"
```

### 2. Schema Drift Monitor (`scripts/schema_drift_monitor.py`)

Production-ready daily monitoring script with alerting capabilities:

#### Features:
- **Daily Schema Snapshots**: Captures complete database schema state
- **Drift Detection**: Compares current schema with baseline
- **Multi-channel Alerts**: Webhook (Slack/Discord/Teams) and email notifications
- **Historical Tracking**: Maintains schema change history
- **Automated Cleanup**: Removes old snapshots to save disk space

#### Usage:
```bash
# Create baseline snapshot
python scripts/schema_drift_monitor.py --prod-db-url="postgresql://..." --create-baseline

# Run daily check
python scripts/schema_drift_monitor.py --prod-db-url="postgresql://..." --alert-webhook="https://hooks.slack.com/..."

# Compare two snapshots
python scripts/schema_drift_monitor.py --compare-snapshots snapshot1.json snapshot2.json

# Generate schema hash
python scripts/schema_drift_monitor.py --generate-hash
```

### 3. Automated Setup (`scripts/setup_schema_monitoring_cron.sh`)

Automated setup script for production deployment:

#### Features:
- **Cron Job Configuration**: Sets up daily monitoring with customizable schedule
- **Database Validation**: Tests connection and creates baseline
- **Log Management**: Configures log rotation and cleanup
- **Alert Configuration**: Sets up webhook and email alerting

#### Usage:
```bash
# Basic setup with webhook alerts
bash scripts/setup_schema_monitoring_cron.sh \
    --prod-db-url="postgresql://user:pass@localhost/greyhound_prod" \
    --alert-webhook="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

# Full setup with email alerts
bash scripts/setup_schema_monitoring_cron.sh \
    --prod-db-url="postgresql://user:pass@localhost/greyhound_prod" \
    --alert-webhook="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK" \
    --email-smtp-server="smtp.gmail.com" \
    --email-username="alerts@yourdomain.com" \
    --email-password="your-app-password" \
    --email-to="devops@yourdomain.com"
```

### 4. Database Models (`models.py`)

SQLAlchemy models that define the expected schema structure:

#### Features:
- **Complete Schema Definition**: All tables, columns, indexes, and foreign keys
- **Alembic Integration**: Used by Alembic for migration generation
- **Index Specifications**: Explicit index definitions for performance
- **Foreign Key Constraints**: Proper referential integrity definitions

## CI/CD Integration

### GitHub Actions Integration

The schema consistency tests are integrated into the CI pipeline (`.github/workflows/ci.yml`):

```yaml
- name: Database Schema Consistency Tests
  run: |
    echo "Running database schema consistency tests..."
    
    # Test 1: Alembic schema consistency (empty diff check)
    python -m pytest tests/test_database_schema_consistency.py::test_alembic_schema_consistency -v
    
    # Test 2: Foreign key indexes check
    python -m pytest tests/test_database_schema_consistency.py::test_foreign_key_indexes -v
    
    # Test 3: Data integrity checks
    python -m pytest tests/test_database_schema_consistency.py::test_data_integrity -v
    
    # Test 4: Schema hash generation
    python -m pytest tests/test_database_schema_consistency.py::test_schema_hash_generation -v
```

### Makefile Integration

Added convenient make targets:

```bash
make schema-tests      # Run all schema consistency tests
make schema-baseline   # Create baseline schema snapshot
make schema-monitor    # Run schema drift monitoring manually
```

## Production Deployment

### 1. Initial Setup

```bash
# 1. Deploy the application to production
# 2. Run the setup script
bash scripts/setup_schema_monitoring_cron.sh \
    --prod-db-url="your-production-database-url" \
    --alert-webhook="your-alert-webhook"

# 3. Verify the cron job
crontab -l | grep schema_drift
```

### 2. Daily Monitoring

The system automatically:
1. **Captures Schema**: Takes a snapshot of the current production schema
2. **Compares Changes**: Compares with the baseline schema
3. **Detects Drift**: Identifies any schema changes or data integrity issues
4. **Sends Alerts**: Notifies via webhook and/or email if issues are found
5. **Logs Results**: Maintains detailed logs of all checks
6. **Cleans Up**: Removes old snapshots to prevent disk space issues

### 3. Alert Types

#### Schema Drift Alerts
Triggered when:
- Schema hash changes
- Tables are added/removed/modified
- Indexes are changed
- Foreign key relationships change

#### Critical Issue Alerts
Triggered when:
- Schema doesn't match models (Alembic detects changes)
- Foreign keys lack indexes
- Data integrity violations found

#### Error Alerts
Triggered when:
- Monitoring script fails
- Database connection issues
- Unexpected errors occur

## Test Categories

### 1. Alembic Schema Consistency Test

**Purpose**: Ensures database schema exactly matches SQLAlchemy models

**How it works**:
1. Runs `alembic revision --autogenerate --compare-type`
2. Analyzes generated migration file
3. Fails if any schema changes are detected

**Fixes schema drift by ensuring**:
- No manual schema changes bypass migrations
- Models and database stay synchronized
- Schema changes go through proper review process

### 2. Foreign Key Index Test

**Purpose**: Ensures all foreign keys have corresponding indexes for performance

**How it works**:
1. Enumerates all foreign key constraints
2. Checks for matching indexes on FK columns
3. Reports missing indexes with suggested CREATE INDEX statements

**Performance benefits**:
- Prevents slow JOIN queries
- Improves referential integrity check speed
- Avoids table lock issues during FK operations

### 3. Data Integrity Test

**Purpose**: Validates data consistency and constraint compliance

**Checks performed**:
- **Orphan Records**: FK references to non-existent records
- **NULL Violations**: NULL values in non-nullable columns
- **Constraint Violations**: Data that violates business rules (e.g., invalid position values)

**Data quality assurance**:
- Prevents data corruption
- Ensures business rule compliance
- Maintains referential integrity

### 4. Schema Hash Test

**Purpose**: Enables automated drift detection

**How it works**:
1. Generates deterministic hash of complete schema structure
2. Compares hashes to detect changes
3. Provides fast change detection mechanism

**Drift detection benefits**:
- Immediate notification of schema changes
- Historical change tracking
- Automated compliance monitoring

## Configuration

### Environment Variables

```bash
# Database connection
DATABASE_URL="postgresql://user:pass@localhost/db"

# Alert webhooks (optional)
ALERT_WEBHOOK="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

# Email configuration (optional)
EMAIL_SMTP_SERVER="smtp.gmail.com"
EMAIL_USERNAME="alerts@yourdomain.com"  
EMAIL_PASSWORD="your-app-password"
EMAIL_TO="devops@yourdomain.com"
```

### Cron Schedule

Default: Daily at 6 AM (`0 6 * * *`)

Common alternatives:
- `0 */4 * * *` - Every 4 hours
- `0 2 * * *` - Daily at 2 AM
- `0 6 * * 1` - Weekly on Mondays at 6 AM

## Monitoring and Maintenance

### Log Files

```
logs/
├── schema_drift_YYYYMMDD.log     # Daily monitoring logs
├── schema_drift_cron.log         # Cron execution logs
└── schema_monitoring_setup.log   # Setup logs
```

### Schema Snapshots

```
schema_baseline.json                    # Baseline schema
schema_snapshot_prod_YYYYMMDD_HHMMSS.json  # Daily snapshots
```

### Log Rotation

Automatic log rotation is configured:
- Daily logs: Rotate after 30 days
- Cron logs: Rotate weekly, keep 12 weeks
- Compression enabled for old logs

## Troubleshooting

### Common Issues

#### 1. "Schema drift detected" alert
**Cause**: Database schema has changed
**Resolution**: 
1. Review the changes in the alert details
2. Create proper Alembic migration if changes are intentional
3. Investigate unauthorized changes if not intentional

#### 2. "Missing foreign key indexes" alert
**Cause**: Foreign keys lack performance indexes
**Resolution**:
1. Add suggested indexes from the alert
2. Update models.py to include the indexes
3. Create Alembic migration for the new indexes

#### 3. "Data integrity violations" alert
**Cause**: Data violates constraints or has orphan records
**Resolution**:
1. Review specific violations in alert details
2. Clean up orphan records
3. Fix NULL violations in non-nullable columns
4. Investigate root cause of data issues

#### 4. "Monitoring script failed" alert
**Cause**: Technical issue with monitoring system
**Resolution**:
1. Check database connectivity
2. Verify permissions and credentials
3. Review error logs for specific issues
4. Test script manually

### Manual Testing

```bash
# Test database connection
python -c "from tests.test_database_schema_consistency import DatabaseSchemaConsistencyTester; tester = DatabaseSchemaConsistencyTester('your-db-url'); print(tester.generate_schema_hash())"

# Run individual tests
python tests/test_database_schema_consistency.py --database-url="your-db-url"

# Test alerts
python scripts/schema_drift_monitor.py --prod-db-url="your-db-url" --alert-webhook="your-webhook"
```

## Best Practices

### 1. Schema Changes
- **Always use Alembic migrations** for schema changes
- **Test migrations** in development first
- **Review migration diffs** before applying to production
- **Coordinate schema changes** with application deployments

### 2. Monitoring
- **Monitor alerts regularly** - don't ignore schema drift warnings
- **Review logs periodically** to ensure monitoring is working
- **Test alert channels** to ensure notifications are received
- **Update baseline** after approved schema changes

### 3. Performance
- **Add indexes for foreign keys** before they become performance bottlenecks
- **Monitor query performance** after schema changes
- **Use database-specific optimizations** where appropriate

### 4. Security
- **Protect database credentials** used in monitoring
- **Limit database permissions** for monitoring user
- **Secure alert channels** to prevent information leakage
- **Review access logs** for unauthorized schema changes

## Integration with Existing Systems

### MLflow Integration
The schema monitoring system works alongside MLflow model registry:
- Model metadata tables are included in monitoring
- Schema changes to model registry are detected
- Model deployment can be gated on schema consistency

### Monitoring Stack Integration
Compatible with common monitoring solutions:
- **Prometheus**: Metrics can be exported for monitoring
- **Grafana**: Dashboards can display schema health
- **DataDog**: Custom metrics and alerts integration
- **New Relic**: Application performance correlation

### CI/CD Pipeline Integration
Seamlessly integrates with:
- **GitHub Actions**: Automated testing in CI
- **Jenkins**: Pipeline integration available
- **GitLab CI**: Compatible with GitLab runners
- **CircleCI**: Standard pytest integration

This comprehensive schema consistency system ensures database reliability, performance, and integrity for the Greyhound Racing Predictor application.
