"""merge conflicting migration heads

Revision ID: 1419b2b82095
Revises: 81268533d929, 9f1a2b3c4d5e
Create Date: 2025-09-01 19:51:19.136781

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1419b2b82095"
down_revision: Union[str, Sequence[str], None] = ("81268533d929", "9f1a2b3c4d5e")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
