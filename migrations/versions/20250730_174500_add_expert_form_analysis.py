"""Add expert form analysis table for PDF integration

Revision ID: add_expert_form_analysis  
Revises: add_fasttrack_schema
Create Date: 2025-07-30 17:45:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = 'add_expert_form_analysis'
down_revision = 'add_fasttrack_schema'
branch_labels = None
depends_on = None


def upgrade():
    """
    Creates the expert_form_analysis table for storing extracted data from 
    FastTrack expert analysis PDFs.
    
    This enables hybrid prediction models that combine algorithmic predictions
    with expert human insights, providing richer analysis and explanations.
    """
    
    # ===============================
    # Create expert_form_analysis
    # ===============================
    op.create_table('expert_form_analysis',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('race_id', sa.Integer(), nullable=False),
        sa.Column('pdf_url', sa.Text(), nullable=True),
        sa.Column('analysis_text', sa.Text(), nullable=True),
        sa.Column('expert_selections', sa.Text(), nullable=True),  # JSON stored as TEXT in SQLite
        sa.Column('confidence_ratings', sa.Text(), nullable=True),  # JSON stored as TEXT in SQLite
        sa.Column('key_insights', sa.Text(), nullable=True),  # JSON stored as TEXT in SQLite
        sa.Column('analysis_date', sa.DateTime(), nullable=True),
        sa.Column('expert_name', sa.Text(), nullable=True),
        sa.Column('extraction_timestamp', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('extraction_confidence', sa.Real(), nullable=True),
        sa.Column('data_source', sa.Text(), nullable=True, server_default='fasttrack_expert'),
        sa.Column('processing_status', sa.Text(), nullable=True, server_default='pending'),  # pending, processed, failed
        sa.Column('processing_notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create foreign key constraint to races table
    # Note: This assumes a races table exists. Adjust the referenced table/column as needed.
    op.create_foreign_key(
        'fk_expert_form_analysis_race_id', 'expert_form_analysis', 'races',
        ['race_id'], ['id'], ondelete='CASCADE'
    )
    
    # Create indexes for performance
    op.create_index('idx_expert_form_analysis_race_id', 'expert_form_analysis', ['race_id'])
    op.create_index('idx_expert_form_analysis_expert_name', 'expert_form_analysis', ['expert_name'])
    op.create_index('idx_expert_form_analysis_analysis_date', 'expert_form_analysis', ['analysis_date'])
    op.create_index('idx_expert_form_analysis_processing_status', 'expert_form_analysis', ['processing_status'])


def downgrade():
    """
    Removes the expert_form_analysis table.
    
    WARNING: This will permanently delete all expert analysis data.
    Ensure you have backups before running this downgrade.
    """
    op.drop_table('expert_form_analysis')


# =====================================
# Helper functions for expert analysis processing
# =====================================

def process_expert_pdf_sample():
    """
    Sample function showing how expert PDF processing would work.
    This is for documentation purposes and should not be executed in migration.
    """
    # This is a placeholder showing the intended workflow:
    
    # 1. Download PDF from FastTrack
    # pdf_content = download_pdf(pdf_url)
    
    # 2. Extract text using OCR/PDF parsing
    # text_content = extract_text_from_pdf(pdf_content)
    
    # 3. Use NLP to identify structured data
    # selections = extract_expert_selections(text_content)
    # insights = extract_key_insights(text_content)
    # confidence = calculate_extraction_confidence(text_content)
    
    # 4. Store in database
    # op.execute("""
    #     INSERT INTO expert_form_analysis (
    #         race_id, pdf_url, analysis_text, expert_selections, 
    #         key_insights, extraction_confidence, processing_status
    #     ) VALUES (?, ?, ?, ?, ?, ?, 'processed')
    # """, (race_id, pdf_url, text_content, json.dumps(selections), 
    #       json.dumps(insights), confidence))
    
    pass


# Sample expert analysis data structure for reference:
SAMPLE_EXPERT_SELECTIONS = {
    "win": [1, 4],  # Top win selections
    "place": [1, 4, 6],  # Place selections
    "quinella": [[1, 4], [1, 6]],  # Quinella combinations
    "trifecta": [[1, 4, 6]],  # Trifecta combinations
    "exotic_bets": {
        "first_four": [[1, 4, 6, 8]],
        "running_double": [1, 3]
    }
}

SAMPLE_CONFIDENCE_RATINGS = {
    "1": 0.85,  # Dog 1 confidence
    "4": 0.75,  # Dog 4 confidence  
    "6": 0.60,  # Dog 6 confidence
    "overall": 0.78  # Overall analysis confidence
}

SAMPLE_KEY_INSIGHTS = {
    "track_bias": "inside boxes favored due to weather conditions",
    "pace": "moderate early pace expected",
    "key_runners": {
        "1": "strong early pace, should lead",
        "4": "best chance from wide draw",
        "6": "value chance if pace is on"
    },
    "weather_impact": "track playing fair despite morning rain",
    "betting_strategy": "quinella 1-4 represents best value"
}
