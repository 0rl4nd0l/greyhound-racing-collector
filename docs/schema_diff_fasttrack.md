# ER Diagram and Schema Diff for FastTrack Data Integration

This document outlines the required changes to the unified database schema to incorporate data from the FastTrack data source.

## ER Diagram Diff

Below is a simplified text-based ER diagram showing the proposed additions and modifications to the existing schema. The new tables are prefixed with `ft_` to denote their origin from FastTrack.

```mermaid
graph TD
    subgraph Existing Schema
        dogs["dogs (master list)"]
        dog_performances["dog_performances (one per race)"]
        races["races (metadata)"]
    end

    subgraph FastTrack Extension
        dogs_ft_extra["dogs_ft_extra (1-to-1 with dogs)"]
        races_ft_extra["races_ft_extra (1-to-1 with races)"]
        dog_performance_ft_extra["dog_performance_ft_extra (1-to-1 with dog_performances)"]
    end

    dogs --o| "extends" |..o dogs_ft_extra
    races --o| "extends" |..o races_ft_extra
    dog_performances --o| "extends" |..o dog_performance_ft_extra

    races --|> dog_performances
    dogs --|> dog_performances
```

**Key:**
- `-->` represents a one-to-many relationship.
- `--o|...o|--` represents a one-to-one relationship for schema extension.

## Proposed Schema Changes

### 1. New Table: `dogs_ft_extra`

This table will store additional, relatively static greyhound attributes available from FastTrack that are not in the current `dogs` table.

**Relationship:** One-to-one with the existing `dogs` table.
- `dogs_ft_extra.dog_id` -> `dogs.id`

**Columns:**
| Column Name | Data Type | Description | Sample Value |
|---|---|---|---|
| `id` | INTEGER | Primary Key | 1 |
| `dog_id` | INTEGER | Foreign Key to `dogs.id` (Unique) | 12345 |
| `sire_name` | TEXT | Sire's (father's) name | Fernando Bale |
| `sire_id` | TEXT | FastTrack ID of the sire | 78910 |
| `dam_name` | TEXT | Dam's (mother's) name | Shining Star |
| `dam_id` | TEXT | FastTrack ID of the dam | 11121 |
| `whelping_date` | DATE | Date of birth for the greyhound | 2021-05-15 |
| `age_days` | INTEGER | Calculated age in days at the time of the last update | 1150 |
| `color`| TEXT | Color of the greyhound | Black |
| `sex` | TEXT | Sex of the greyhound (Dog/Bitch) | Dog |
| `ear_brand` | TEXT | Unique ear brand identifier | VABCD |
| `career_starts` | INTEGER | Total number of official starts | 25 |
| `career_wins` | INTEGER | Total number of wins | 10 |
| `career_places` | INTEGER | Total number of places (2nd/3rd) | 5 |
| `career_win_percent`| REAL | Win percentage over the career | 40.0 |
| `winning_boxes_json`| JSON | JSON object detailing wins per box number | `{"1": 3, "4": 2, "8": 5}` |

### 2. New Table: `races_ft_extra`

This table stores supplementary race-specific details from FastTrack that don't fit into the existing `races` table.

**Relationship:** One-to-one with the existing `races` table.
- `races_ft_extra.race_id` -> `races.id`

**Columns:**
| Column Name | Data Type | Description | Sample Value |
|---|---|---|---|
| `id` | INTEGER | Primary Key | 1 |
| `race_id` | INTEGER | Foreign Key to `races.id` (Unique) | 67890 |
| `track_rating`| TEXT | Official track rating (e.g., Good, Soft) | Good |
| `weather_description`| TEXT | Detailed weather conditions from the track | Fine, 18°C, Light Breeze |
| `race_comment` | TEXT | Official steward or commentator summary for the race | "Clear run for all." |
| `split_1_time_winner` | REAL | Sectional time for the winner at the first marker | 5.42 |
| `split_2_time_winner` | REAL | Sectional time for the winner at the second marker | 17.88 |
| `run_home_time_winner`| REAL | Calculated run-home time for the winner | 11.64 |
| `video_url` | TEXT | URL to the official race replay video | `https://...` |
| `stewards_report_url` | TEXT | URL to the official stewards' report | `https://...` |

### 3. New Table: `dog_performance_ft_extra`

This table extends the `dog_performances` table with highly granular, performance-specific data from FastTrack for each dog in a race.

**Relationship:** One-to-one with the existing `dog_performances` table.
- `dog_performance_ft_extra.performance_id` -> `dog_performances.id`

**Columns:**
| Column Name | Data Type | Description | Sample Value |
|---|---|---|---|
| `id` | INTEGER | Primary Key | 1 |
| `performance_id`| INTEGER | Foreign Key to `dog_performances.id` (Unique)| 98765 |
| `pir_rating` | TEXT | Position In Running (PIR) code, e.g., 'M/432'| M/432 |
| `split_1_time` | REAL | The dog's individual first sectional time | 5.51 |
| `split_2_time` | REAL | The dog's individual second sectional time | 18.05 |
| `run_home_time`| REAL | The dog's individual run-home time | 11.70 |
| `beaten_margin` | REAL | Official margin from the winner in lengths | 2.75 |
| `in_race_comment` | TEXT | Commentator's description of the dog's performance | "Ran on well" |
| `fixed_odds_sp` | REAL | Starting price from the fixed odds market | 4.50 |
| `prize_money` | REAL | Prize money won by the dog in this race | 1500.00 |
| `pre_race_weight` | REAL | The dog's official weight before the race | 32.5 |

## Expert Form Analysis PDFs Integration

FastTrack provides expert form analysis PDFs for each race meeting that contain professional insights, predictions, and commentary. These PDFs represent valuable human expertise that can significantly enhance our predictive capabilities.

### Proposed Extension: `expert_form_analysis` Table

To fully leverage the expert analysis PDFs, we propose an additional table:

**Relationship:** Many-to-one with the `races` table (multiple expert analyses can exist per race).
- `expert_form_analysis.race_id` -> `races.id`

**Columns:**
| Column Name | Data Type | Description | Sample Value |
|---|---|---|---|
| `id` | INTEGER | Primary Key | 1 |
| `race_id` | INTEGER | Foreign Key to `races.id` | 67890 |
| `pdf_url` | TEXT | Direct URL to the expert analysis PDF | `https://fasttrack.grv.org.au/expert/analysis/race123.pdf` |
| `analysis_text` | TEXT | Extracted text content from the PDF | "Trap 1 looks the goods here..." |
| `expert_selections` | JSON | Structured expert picks | `{"win": [1, 4], "place": [1, 4, 6], "quinella": [[1,4]]}` |
| `confidence_ratings` | JSON | Expert confidence scores per selection | `{"1": 0.85, "4": 0.75, "6": 0.60}` |
| `key_insights` | JSON | Extracted insights and reasoning | `{"track_bias": "inside boxes favored", "pace": "moderate"}` |
| `analysis_date` | DATETIME | When the expert analysis was published | 2025-07-30 15:30:00 |
| `expert_name` | TEXT | Name/ID of the expert analyst | John Smith |
| `extraction_timestamp` | DATETIME | When we processed the PDF | 2025-07-30 16:45:00 |
| `extraction_confidence` | REAL | Confidence in our text extraction (0-1) | 0.92 |

### Enhanced Analysis Capabilities

With expert form analysis integration, our system would provide:

1. **Hybrid Predictions**: Combine data-driven ML predictions with expert human insights
2. **Expert Tracking**: Monitor expert prediction accuracy over time to weight their opinions
3. **Reasoning Context**: Provide users with not just predictions, but explanations from experts
4. **Consensus Analysis**: Compare multiple expert opinions when available
5. **Performance Validation**: Validate our ML models against expert selections

### Implementation Strategy

1. **PDF Scraping**: Automated download of expert analysis PDFs from FastTrack
2. **Text Extraction**: Use OCR/PDF parsing to extract structured data
3. **NLP Processing**: Apply natural language processing to identify:
   - Dog selections and confidence levels
   - Key insights about track conditions, pace, bias
   - Reasoning behind selections
4. **Data Integration**: Merge expert insights with existing race data
5. **Predictive Enhancement**: Use expert data as additional features in ML models

### Sample Expert Analysis Integration

```python
# Example of how expert analysis would enhance predictions
def enhanced_race_prediction(race_id):
    # Get base ML prediction
    ml_prediction = get_ml_prediction(race_id)
    
    # Get expert analysis
    expert_analysis = get_expert_analysis(race_id)
    
    # Combine predictions with weighting based on expert accuracy
    if expert_analysis:
        expert_weight = calculate_expert_weight(expert_analysis.expert_name)
        combined_prediction = blend_predictions(
            ml_prediction, 
            expert_analysis.expert_selections,
            expert_weight
        )
        
        return {
            'prediction': combined_prediction,
            'ml_confidence': ml_prediction.confidence,
            'expert_insights': expert_analysis.key_insights,
            'reasoning': expert_analysis.analysis_text
        }
    
    return ml_prediction
```

## Rationale

- **Separation of Concerns:** By creating new tables prefixed with `ft_`, we keep the FastTrack-specific data separate from the core unified schema. This makes it easier to manage, update, and even remove if the data source becomes unavailable, without impacting the primary application logic.
- **Performance:** One-to-one relationships avoid bloating the main tables (`dogs`, `races`, `dog_performances`) with dozens of new columns, many of which may be null or not always available. This keeps the core tables lean and fast for frequent queries.
- **Scalability:** This extension model allows for adding other data sources in the future (e.g., `_source2_extra`) without requiring constant modification of the core tables.
- **Data Integrity:** Foreign key constraints ensure that the extra data is always linked to a valid record in the main tables, preventing orphaned data.
- **Human-AI Collaboration:** Expert form analysis bridges the gap between pure algorithmic predictions and human expertise, creating a more robust and explainable prediction system.

