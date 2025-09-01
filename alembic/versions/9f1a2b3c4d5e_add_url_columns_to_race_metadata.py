"""Ensure url and sportsbet_url columns exist on race_metadata

Revision ID: 9f1a2b3c4d5e
Revises: bdd69f3b1271
Create Date: 2025-08-30 08:50:00.000000

This migration is idempotent: it only adds missing columns if they don't exist.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "9f1a2b3c4d5e"
down_revision: Union[str, Sequence[str], None] = "bdd69f3b1271"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    try:
        cols = [c.get("name") for c in inspector.get_columns(table_name)]
        return column_name in cols
    except Exception:
        return False


def upgrade() -> None:
    # Add url if missing
    if not _has_column("race_metadata", "url"):
        try:
            op.add_column("race_metadata", sa.Column("url", sa.String(), nullable=True))
        except Exception:
            pass
    # Add sportsbet_url if missing
    if not _has_column("race_metadata", "sportsbet_url"):
        try:
            op.add_column("race_metadata", sa.Column("sportsbet_url", sa.String(), nullable=True))
        except Exception:
            pass


def downgrade() -> None:
    # Drop columns if present
    if _has_column("race_metadata", "sportsbet_url"):
        try:
            op.drop_column("race_metadata", "sportsbet_url")
        except Exception:
            pass
    if _has_column("race_metadata", "url"):
        try:
            op.drop_column("race_metadata", "url")
        except Exception:
            pass

