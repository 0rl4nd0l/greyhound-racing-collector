import logging
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DATABASE_PATH = "greyhound_racing_data.db"


def apply_schema():
    """
    Applies the FastTrack and Expert Form Analysis schema extensions to the database.
    This script is designed for one-time setup when Alembic is not in use.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        logging.info(f"Successfully connected to {DATABASE_PATH}")

        # --- Schema from 20250730_173000_add_fasttrack_schema.py ---
        logging.info("Applying schema: dogs_ft_extra")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dogs_ft_extra (
                id INTEGER NOT NULL PRIMARY KEY,
                dog_id INTEGER NOT NULL UNIQUE,
                sire_name TEXT,
                sire_id TEXT,
                dam_name TEXT,
                dam_id TEXT,
                whelping_date DATE,
                age_days INTEGER,
                color TEXT,
                sex TEXT,
                ear_brand TEXT,
                career_starts INTEGER,
                career_wins INTEGER,
                career_places INTEGER,
                career_win_percent REAL,
                winning_boxes_json TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'fasttrack',
                FOREIGN KEY(dog_id) REFERENCES dogs(id) ON DELETE CASCADE
            )
        """
        )

        logging.info("Applying schema: races_ft_extra")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS races_ft_extra (
                id INTEGER NOT NULL PRIMARY KEY,
                race_id INTEGER NOT NULL UNIQUE,
                track_rating TEXT,
                weather_description TEXT,
                race_comment TEXT,
                split_1_time_winner REAL,
                split_2_time_winner REAL,
                run_home_time_winner REAL,
                video_url TEXT,
                stewards_report_url TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'fasttrack',
                FOREIGN KEY(race_id) REFERENCES races(id) ON DELETE CASCADE
            )
        """
        )

        logging.info("Applying schema: dog_performance_ft_extra")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dog_performance_ft_extra (
                id INTEGER NOT NULL PRIMARY KEY,
                performance_id INTEGER NOT NULL UNIQUE,
                pir_rating TEXT,
                split_1_time REAL,
                split_2_time REAL,
                run_home_time REAL,
                beaten_margin REAL,
                in_race_comment TEXT,
                fixed_odds_sp REAL,
                prize_money REAL,
                pre_race_weight REAL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'fasttrack',
                FOREIGN KEY(performance_id) REFERENCES dog_performances(id) ON DELETE CASCADE
            )
        """
        )
        logging.info("FastTrack schema applied successfully.")

        # --- Schema from 20250730_174500_add_expert_form_analysis.py ---
        logging.info("Applying schema: expert_form_analysis")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS expert_form_analysis (
                id INTEGER NOT NULL PRIMARY KEY,
                race_id INTEGER NOT NULL,
                pdf_url TEXT,
                analysis_text TEXT,
                expert_selections TEXT,
                confidence_ratings TEXT,
                key_insights TEXT,
                analysis_date DATETIME,
                expert_name TEXT,
                extraction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                extraction_confidence REAL,
                data_source TEXT DEFAULT 'fasttrack_expert',
                processing_status TEXT DEFAULT 'pending',
                processing_notes TEXT,
                FOREIGN KEY(race_id) REFERENCES races(id) ON DELETE CASCADE
            )
        """
        )
        logging.info("Expert Form Analysis schema applied successfully.")

        conn.commit()
        logging.info("Schema changes committed.")

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")


if __name__ == "__main__":
    apply_schema()
