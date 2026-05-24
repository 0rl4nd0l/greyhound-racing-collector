"""Add start_datetime column to race_metadata (Alembic)

Revision ID: a90f1d2c3b4e
Revises: 1419b2b82095
Create Date: 2025-09-11 07:59:10.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "a90f1d2c3b4e"
down_revision: Union[str, Sequence[str], None] = "1419b2b82095"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add start_datetime to race_metadata if not present
    with op.batch_alter_table("race_metadata") as batch_op:
        batch_op.add_column(sa.Column("start_datetime", sa.DateTime(), nullable=True))


def downgrade() -> None:
    # Remove start_datetime from race_metadata
    with op.batch_alter_table("race_metadata") as batch_op:
        batch_op.drop_column("start_datetime")

