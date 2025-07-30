# Database Migrations with Alembic

This directory contains Alembic database migration scripts for the greyhound racing collector system.

## Setup

Alembic is already configured and ready to use. The configuration is in `alembic.ini` and points to the main SQLite database `greyhound_racing_data.db`.

## Common Commands

### Check Current Migration Status
```bash
alembic current
```

### View Migration History
```bash
alembic history --verbose
```

### Create a New Migration
```bash
# Auto-generate migration (recommended)
alembic revision --autogenerate -m "Description of changes"

# Manual migration (if autogenerate doesn't work)
alembic revision -m "Description of changes"
```

### Apply Migrations
```bash
# Upgrade to latest version
alembic upgrade head

# Upgrade to specific revision
alembic upgrade [revision_id]

# Downgrade to previous version
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade [revision_id]
```

### View Migration Details
```bash
# Show the SQL that would be executed (dry run)
alembic upgrade head --sql

# Show specific migration info
alembic show [revision_id]
```

## Migration Files

Migration files are stored in `alembic/versions/` and follow the naming pattern:
`[revision_id]_[description].py`

Each migration file contains:
- `upgrade()` function: Apply the migration
- `downgrade()` function: Reverse the migration

## Database Models

The SQLAlchemy models are defined in `models/database_models.py` and are automatically imported by Alembic for autogeneration.

## Best Practices

1. **Always review generated migrations** before applying them
2. **Test migrations on a copy of production data** before applying to production
3. **Keep migrations small and focused** on a single change
4. **Add descriptive messages** to migration files
5. **Backup your database** before running migrations on production

## Troubleshooting

### Autogenerate Issues
If `--autogenerate` fails due to foreign key constraints or complex schema changes, create a manual migration instead:

```bash
alembic revision -m "Manual migration description"
```

Then edit the generated file to add the necessary operations.

### Existing Database
For an existing database (like our current setup), we've created a baseline migration that represents the current state. Future changes should be added as new migrations.

### Index Conflicts
Some operations may fail if database objects already exist. Handle this gracefully in migration code:

```python
def upgrade():
    try:
        op.create_index('index_name', 'table_name', ['column_name'])
    except Exception:
        pass  # Index already exists
```

## Examples

Our migrations include:
- `bdd69f3b1271`: Baseline schema for existing database
- `9860d6e5a183`: Added performance indexes

Future migrations might include:
- Adding new tables for features
- Adding/removing columns
- Data type changes
- Constraint modifications
