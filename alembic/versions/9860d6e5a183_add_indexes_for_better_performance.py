"""Add indexes for better performance

Revision ID: 9860d6e5a183
Revises: bdd69f3b1271
Create Date: 2025-07-30 20:37:58.150572

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9860d6e5a183"
down_revision: Union[str, Sequence[str], None] = "bdd69f3b1271"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Add database indexes for better query performance."""
    # Add indexes for commonly queried columns (with existence checks)
    try:
        op.create_index(
            "idx_race_metadata_venue_date", "race_metadata", ["venue", "race_date"]
        )
    except Exception:
        pass  # Index already exists

    try:
        op.create_index("idx_race_metadata_race_id", "race_metadata", ["race_id"])
    except Exception:
        pass  # Index already exists

    try:
        op.create_index("idx_dog_race_data_race_id", "dog_race_data", ["race_id"])
    except Exception:
        pass  # Index already exists

    try:
        op.create_index(
            "idx_dog_race_data_dog_name", "dog_race_data", ["dog_clean_name"]
        )
    except Exception:
        pass  # Index already exists

    try:
        op.create_index("idx_dogs_clean_name", "dogs", ["clean_name"])
    except Exception:
        pass  # Index already exists

    try:
        op.create_index("idx_dogs_trainer", "dogs", ["trainer"])
    except Exception:
        pass  # Index already exists


def downgrade() -> None:
    """Downgrade schema - Remove the added indexes."""
    op.drop_index("idx_dogs_trainer", table_name="dogs")
    op.drop_index("idx_dogs_clean_name", table_name="dogs")
    op.drop_index("idx_dog_race_data_dog_name", table_name="dog_race_data")
    op.drop_index("idx_dog_race_data_race_id", table_name="dog_race_data")
    op.drop_index("idx_race_metadata_race_id", table_name="race_metadata")
    op.drop_index("idx_race_metadata_venue_date", table_name="race_metadata")
