"""add_foreign_keys_to_race_id

Revision ID: 000000000003
Revises: 000000000002
Create Date: 2025-08-01 00:30:00.000000

"""

import sqlalchemy as sa

from alembic import op

depends_on = None

revision = "000000000003"
down_revision = "000000000002"
branch_labels = None
depends_on = None


def upgrade():
    """Apply foreign keys on race_id to ensure data integrity."""
    with op.batch_alter_table("dog_race_data") as batch_op:
        batch_op.create_foreign_key(
            "fk_dog_race_data_race_id",
            "race_metadata",
            ["race_id"],
            ["race_id"],
            ondelete="CASCADE",
        )

    with op.batch_alter_table("enhanced_expert_data") as batch_op:
        batch_op.create_foreign_key(
            "fk_enhanced_expert_data_race_id",
            "race_metadata",
            ["race_id"],
            ["race_id"],
            ondelete="CASCADE",
        )

    # Repeat for other tables using race_id


def downgrade():
    """Remove foreign keys from race_id if a rollback is needed."""
    with op.batch_alter_table("dog_race_data") as batch_op:
        batch_op.drop_constraint("fk_dog_race_data_race_id", type_="foreignkey")

    with op.batch_alter_table("enhanced_expert_data") as batch_op:
        batch_op.drop_constraint("fk_enhanced_expert_data_race_id", type_="foreignkey")

    # Repeat for other tables using race_id
