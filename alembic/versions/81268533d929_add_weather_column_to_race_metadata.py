"""add weather column to race_metadata

Revision ID: 81268533d929
Revises: 8d202048814f
Create Date: 2025-08-02 14:39:24.369021

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '81268533d929'
down_revision: Union[str, Sequence[str], None] = '8d202048814f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('race_metadata', sa.Column('weather', sa.String(), nullable=True))

    # Populate the new 'weather' column with data from 'weather_condition'
    race_metadata = sa.table('race_metadata',
        sa.column('weather', sa.String()),
        sa.column('weather_condition', sa.String())
    )
    op.execute(
        race_metadata.update().values(
            weather=sa.text('weather_condition')
        )
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('race_metadata', 'weather')
