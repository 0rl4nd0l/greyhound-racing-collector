"""Add FastTrack schema extension

Revision ID: add_fasttrack_schema
Revises: initial_migration
Create Date: 2025-07-30 17:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic
revision = 'add_fasttrack_schema'
down_revision = 'initial_migration'  # Update this to match your actual current migration
branch_labels = None
depends_on = None


def upgrade():
    """
    Creates FastTrack extension tables to supplement the existing unified schema.
    
    This migration adds three new tables:
    1. dogs_ft_extra: Additional dog-specific data from FastTrack
    2. races_ft_extra: Enhanced race metadata from FastTrack  
    3. dog_performance_ft_extra: Detailed performance metrics per dog per race
    
    All three tables have one-to-one relationships with their corresponding
    base tables (dogs, races, dog_performances).
    """
    
    # =====================
    # Create dogs_ft_extra
    # =====================
    op.create_table('dogs_ft_extra',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('dog_id', sa.Integer(), nullable=False),
        sa.Column('sire_name', sa.Text(), nullable=True),
        sa.Column('sire_id', sa.Text(), nullable=True),
        sa.Column('dam_name', sa.Text(), nullable=True), 
        sa.Column('dam_id', sa.Text(), nullable=True),
        sa.Column('whelping_date', sa.Date(), nullable=True),
        sa.Column('age_days', sa.Integer(), nullable=True),
        sa.Column('color', sa.Text(), nullable=True),
        sa.Column('sex', sa.Text(), nullable=True),
        sa.Column('ear_brand', sa.Text(), nullable=True),
        sa.Column('career_starts', sa.Integer(), nullable=True),
        sa.Column('career_wins', sa.Integer(), nullable=True),
        sa.Column('career_places', sa.Integer(), nullable=True),
        sa.Column('career_win_percent', sa.Real(), nullable=True),
        sa.Column('winning_boxes_json', sa.Text(), nullable=True),  # JSON stored as TEXT in SQLite
        sa.Column('last_updated', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('data_source', sa.Text(), nullable=True, server_default='fasttrack'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('dog_id', name='uq_dogs_ft_extra_dog_id')
    )
    
    # Create foreign key constraint to dogs table
    # Note: This assumes a dogs table exists. Adjust the referenced table/column as needed.
    op.create_foreign_key(
        'fk_dogs_ft_extra_dog_id', 'dogs_ft_extra', 'dogs', 
        ['dog_id'], ['id'], ondelete='CASCADE'
    )
    
    # Create indexes for performance
    op.create_index('idx_dogs_ft_extra_dog_id', 'dogs_ft_extra', ['dog_id'])
    op.create_index('idx_dogs_ft_extra_sire_dam', 'dogs_ft_extra', ['sire_name', 'dam_name'])
    
    # =====================
    # Create races_ft_extra
    # =====================  
    op.create_table('races_ft_extra',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('race_id', sa.Integer(), nullable=False),
        sa.Column('track_rating', sa.Text(), nullable=True),
        sa.Column('weather_description', sa.Text(), nullable=True),
        sa.Column('race_comment', sa.Text(), nullable=True),
        sa.Column('split_1_time_winner', sa.Real(), nullable=True),
        sa.Column('split_2_time_winner', sa.Real(), nullable=True),
        sa.Column('run_home_time_winner', sa.Real(), nullable=True),
        sa.Column('video_url', sa.Text(), nullable=True),
        sa.Column('stewards_report_url', sa.Text(), nullable=True),
        sa.Column('last_updated', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('data_source', sa.Text(), nullable=True, server_default='fasttrack'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('race_id', name='uq_races_ft_extra_race_id')
    )
    
    # Create foreign key constraint to races table  
    # Note: This assumes a races table exists. Adjust the referenced table/column as needed.
    op.create_foreign_key(
        'fk_races_ft_extra_race_id', 'races_ft_extra', 'races',
        ['race_id'], ['id'], ondelete='CASCADE'  
    )
    
    # Create indexes for performance
    op.create_index('idx_races_ft_extra_race_id', 'races_ft_extra', ['race_id'])
    op.create_index('idx_races_ft_extra_track_rating', 'races_ft_extra', ['track_rating'])
    
    # ===============================
    # Create dog_performance_ft_extra  
    # ===============================
    op.create_table('dog_performance_ft_extra',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('performance_id', sa.Integer(), nullable=False),
        sa.Column('pir_rating', sa.Text(), nullable=True),
        sa.Column('split_1_time', sa.Real(), nullable=True),
        sa.Column('split_2_time', sa.Real(), nullable=True),
        sa.Column('run_home_time', sa.Real(), nullable=True),
        sa.Column('beaten_margin', sa.Real(), nullable=True),
        sa.Column('in_race_comment', sa.Text(), nullable=True),
        sa.Column('fixed_odds_sp', sa.Real(), nullable=True),
        sa.Column('prize_money', sa.Real(), nullable=True),
        sa.Column('pre_race_weight', sa.Real(), nullable=True),
        sa.Column('last_updated', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('data_source', sa.Text(), nullable=True, server_default='fasttrack'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('performance_id', name='uq_dog_performance_ft_extra_performance_id')
    )
    
    # Create foreign key constraint to dog_performances table
    # Note: This assumes a dog_performances table exists. Adjust the referenced table/column as needed.
    op.create_foreign_key(
        'fk_dog_performance_ft_extra_performance_id', 'dog_performance_ft_extra', 'dog_performances',
        ['performance_id'], ['id'], ondelete='CASCADE'
    )
    
    # Create indexes for performance
    op.create_index('idx_dog_performance_ft_extra_performance_id', 'dog_performance_ft_extra', ['performance_id'])
    op.create_index('idx_dog_performance_ft_extra_margins', 'dog_performance_ft_extra', ['beaten_margin'])
    op.create_index('idx_dog_performance_ft_extra_sectionals', 'dog_performance_ft_extra', ['split_1_time', 'split_2_time'])


def downgrade():
    """
    Removes FastTrack extension tables.
    
    WARNING: This will permanently delete all FastTrack-specific data.
    Ensure you have backups before running this downgrade.
    """
    
    # Drop tables in reverse order to handle foreign key dependencies
    op.drop_table('dog_performance_ft_extra')
    op.drop_table('races_ft_extra') 
    op.drop_table('dogs_ft_extra')


# =====================================
# Migration validation and data helpers
# =====================================

def validate_base_tables_exist():
    """
    Helper function to validate that required base tables exist before applying migration.
    This should be called manually during deployment planning.
    """
    # This is a placeholder - in practice, you'd check the actual database
    required_tables = ['dogs', 'races', 'dog_performances']
    print("Checking for required base tables:")
    for table in required_tables:
        print(f"  - {table}: (validation needed)")
    print("Please ensure all base tables exist before running this migration.")


def populate_sample_data():
    """
    Helper function to populate sample FastTrack data after migration.
    This is for testing purposes only and should not be used in production.
    """
    # Sample data insertion queries would go here
    # This is commented out to prevent accidental execution
    
    # Example:
    # op.execute("""
    #     INSERT INTO dogs_ft_extra (dog_id, sire_name, dam_name, career_starts, career_wins)
    #     VALUES (1, 'Fernando Bale', 'Shining Star', 25, 10)
    # """)
    
    pass


# =====================================
# Expert Form Analysis Integration
# =====================================

"""
EXPERT FORM ANALYSIS INTEGRATION NOTES:

The FastTrack website provides expert form analysis PDFs for each race that contain
valuable insights and predictions. To fully leverage this data, consider extending
the schema further with:

1. Additional table: expert_form_analysis
   - race_id (FK to races)
   - pdf_url (URL to the expert analysis PDF)
   - analysis_text (extracted text content)
   - key_selections (JSON with expert picks)
   - confidence_ratings (JSON with confidence scores)
   - analysis_date (when the analysis was published)
   
2. Integration with ML pipeline:
   - Extract structured data from PDF analysis using OCR/text parsing
   - Compare expert predictions with actual race outcomes
   - Use expert insights as additional features for ML models
   - Track expert prediction accuracy over time

3. Enhanced prediction capability:
   - Combine traditional data-driven predictions with expert insights
   - Weight expert opinions based on historical accuracy
   - Provide users with both algorithmic and expert-based recommendations

This would create a truly comprehensive analysis system combining:
- Historical race data (current system)
- Detailed FastTrack metrics (this migration)
- Expert human analysis (future enhancement)
"""
