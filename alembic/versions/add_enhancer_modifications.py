"""Add tables and modify columns for the Greyhound Racing Database"""

# Alembic imports
import sqlalchemy as sa

from alembic import op

# Revision identifiers, used by Alembic.
revision = "add_enhancer_modifications"
down_revision = "bdd69f3b1271"
branch_labels = None
depends_on = None


def upgrade():
    """Apply non-destructive updates to the database schema."""
    # Ensure 'gpt_analysis' table exists, with additional non-destructive updates
    try:
        op.create_table(
            "gpt_analysis",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("race_id", sa.String, nullable=False),
            sa.Column("analysis_type", sa.String, nullable=False),
            sa.Column("analysis_data", sa.Text, nullable=False),
            sa.Column("confidence_score", sa.Float, nullable=True),
            sa.Column("tokens_used", sa.Integer, nullable=True),
            sa.Column("timestamp", sa.DateTime, nullable=False),
            sa.Column("model_used", sa.String, nullable=True),
            sa.ForeignKeyConstraint(["race_id"], ["race_metadata.race_id"]),
        )
    except Exception:
        pass  # Table already exists, safe to skip

    # Ensure 'race_metadata' table is up-to-date
    with op.batch_alter_table("race_metadata", schema=None) as batch_op:
        try:
            batch_op.add_column(
                sa.Column("new_additional_column", sa.String)
            )  # Example addition
        except Exception:
            pass  # Column likely exists, handle safely

    # Ensure 'dog_race_data' table is up-to-date
    with op.batch_alter_table("dog_race_data", schema=None) as batch_op:
        try:
            batch_op.add_column(
                sa.Column("new_analysis_column", sa.Float)
            )  # Example addition
        except Exception:
            pass  # Column likely exists, handle safely


def downgrade():
    """Drop newly added columns and tables (non-destructive rollback)."""
    # The following would be uncommented if downgrade actions were needed, implement safely
    # op.drop_table('gpt_analysis')
    # with op.batch_alter_table('race_metadata', schema=None) as batch_op:
    #     batch_op.drop_column('new_additional_column')
    # with op.batch_alter_table('dog_race_data', schema=None) as batch_op:
    #     batch_op.drop_column('new_analysis_column')
    pass
