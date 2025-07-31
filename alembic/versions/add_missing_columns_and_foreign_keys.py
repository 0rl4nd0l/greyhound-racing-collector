"""add_missing_columns_and_foreign_keys.py

Revision ID: 000000000002
Revises: add_enhancer_modifications
Create Date: 2025-08-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

depends_on = None

revision = '000000000002'
down_revision = 'add_enhancer_modifications'
branch_labels = None
depends_on = None

def upgrade():
    """Non-destructive schema additions and updates."""
    # Add columns to `dog_race_data` table
    with op.batch_alter_table('dog_race_data') as batch_op:
        batch_op.add_column(sa.Column('odds', sa.Text))
        batch_op.add_column(sa.Column('trainer', sa.Text))
        batch_op.add_column(sa.Column('winning_time', sa.Text))
        batch_op.add_column(sa.Column('placing', sa.Integer))
        batch_op.add_column(sa.Column('form', sa.Text))

    # Add indexes
    op.create_index('idx_dog_race_data_race', 'dog_race_data', ['race_id'], unique=False)

    # Add columns to 'dogs' table
    with op.batch_alter_table('dogs') as batch_op:
        batch_op.add_column(sa.Column('weight', sa.Numeric(3, 2)))
        batch_op.add_column(sa.Column('age', sa.Integer))
        batch_op.add_column(sa.Column('id', sa.Integer))
        batch_op.add_column(sa.Column('color', sa.Text))
        batch_op.add_column(sa.Column('owner', sa.Text))
        batch_op.add_column(sa.Column('trainer', sa.Text))
        batch_op.add_column(sa.Column('sex', sa.Text))

    # Reinstate foreign keys
    with op.batch_alter_table('race_analytics') as batch_op:
        # Assuming race_analytics should have an FK on race_id
        batch_op.create_foreign_key(None, 'race_metadata', ['race_id'], ['race_id'], ondelete="CASCADE")


def downgrade():
    """Remove recently added schema changes (non-destructive rollback)."""
    # (Example): Often, without fully removing columns and affecting data integrity.
    with op.batch_alter_table('dog_race_data') as batch_op:
        batch_op.drop_column('odds')
        batch_op.drop_column('trainer')
        batch_op.drop_column('winning_time')
        batch_op.drop_column('placing')
        batch_op.drop_column('form')

    op.drop_index('idx_dog_race_data_race', table_name='dog_race_data')

    with op.batch_alter_table('dogs') as batch_op:
        batch_op.drop_column('weight')
        batch_op.drop_column('age')
        batch_op.drop_column('id')
        batch_op.drop_column('color')
        batch_op.drop_column('owner')
        batch_op.drop_column('trainer')
        batch_op.drop_column('sex')

    with op.batch_alter_table('race_analytics') as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')

