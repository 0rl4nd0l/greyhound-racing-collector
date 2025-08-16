"""merge heads

Revision ID: 8d202048814f
Revises: 9860d6e5a183, 000000000003
Create Date: 2025-08-02 14:39:08.903177

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8d202048814f'
down_revision: Union[str, Sequence[str], None] = ('9860d6e5a183', '000000000003')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
