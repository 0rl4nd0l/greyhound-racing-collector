"""Initial database schema

Revision ID: bdd69f3b1271
Revises: 
Create Date: 2025-07-30 20:36:29.011084

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bdd69f3b1271'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.
    
    This migration represents the baseline schema for the existing
    greyhound_racing_data.db database. No actual changes are made
    as the database already exists with the current schema.
    """
    # No operations needed - this is a baseline migration
    # for an existing database.
    pass


def downgrade() -> None:
    """Downgrade schema.
    
    Cannot downgrade from the initial baseline schema.
    """
    raise NotImplementedError("Cannot downgrade from baseline schema")
