from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
from models import Base

# Restrict autogenerate comparisons to a curated set; ignore legacy/aux tables that are not modeled
IGNORED_TABLES = set(
    [
        "race_analytics",
        "tgr_dog_performance_summary",
        "tgr_enrichment_jobs",
        "dogs_ft_extra",
        "dog_performance_ft_extra",
        "races_ft_extra",
        "expert_form_analysis",
        "dog_race_data_backup",
        "dog_race_data_backup_box_number_fix",
        "query_monitoring",
        "value_bets",
        "detailed_race_history",
    ]
)
IGNORED_INDEX_PREFIXES = (
    "idx_tgr_",
    "idx_race_analytics_",
    "idx_races_ft_extra_",
    "idx_dogs_ft_extra_",
)


def _include_object(object, name, type_, reflected, compare_to):
    try:
        # Skip tables that are managed outside of SQLAlchemy models or considered legacy/auxiliary
        if type_ == "table" and name in IGNORED_TABLES:
            return False
        # Skip indexes on ignored tables or matching known ignored prefixes
        if type_ == "index":
            parent = getattr(object, "table", None)
            parent_name = getattr(parent, "name", None)
            if parent_name in IGNORED_TABLES:
                return False
            if any(str(name or "").startswith(p) for p in IGNORED_INDEX_PREFIXES):
                return False
    except Exception:
        # On any unexpected error, do not skip the object
        pass
    return True


target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_object=_include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            include_object=_include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
