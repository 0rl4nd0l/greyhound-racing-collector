#!/usr/bin/env python3
"""
Create TGR Placeholders
======================

This script quickly creates placeholder TGR records for all dogs in the system
to solve the coverage issue and enable ML training.
"""

import logging
import sqlite3
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_tgr_placeholders(db_path: str = "greyhound_racing_data.db"):
    """Create placeholder TGR data for all dogs in the system."""

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all unique dogs that don't have TGR data yet
        cursor.execute(
            """
            SELECT DISTINCT d.dog_clean_name, COUNT(*) as race_count
            FROM dog_race_data d
            LEFT JOIN tgr_dog_performance_summary t ON d.dog_clean_name = t.dog_name
            WHERE d.dog_clean_name IS NOT NULL 
              AND d.dog_clean_name != ''
              AND t.dog_name IS NULL
            GROUP BY d.dog_clean_name
            ORDER BY race_count DESC
        """
        )

        dogs = cursor.fetchall()
        logger.info(f"Found {len(dogs)} dogs without TGR data")

        # Create placeholder records
        placeholder_data = (
            '{"data_source": "placeholder", "processing_status": "pending_tgr"}'
        )
        current_time = datetime.now().isoformat()

        success_count = 0
        for dog_name, race_count in dogs:
            try:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO tgr_dog_performance_summary
                    (dog_name, performance_data, last_updated, total_entries,
                     wins, places, win_percentage, place_percentage, average_position,
                     best_position, consistency_score, form_trend, distance_versatility,
                     venues_raced)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        dog_name,
                        placeholder_data,
                        current_time,
                        0,  # total_entries
                        0,  # wins
                        0,  # places
                        0.0,  # win_percentage
                        0.0,  # place_percentage
                        8.0,  # average_position (worst position)
                        8,  # best_position
                        0.0,  # consistency_score
                        "unknown",  # form_trend
                        0,  # distance_versatility
                        0,  # venues_raced
                    ],
                )
                success_count += 1

                if success_count % 100 == 0:
                    logger.info(f"Created {success_count} placeholder records...")

            except Exception as e:
                logger.error(f"Error creating placeholder for {dog_name}: {e}")

        conn.commit()
        conn.close()

        logger.info(f"✅ Created {success_count} TGR placeholder records")

        # Show updated stats
        show_tgr_stats(db_path)

        return success_count

    except Exception as e:
        logger.error(f"Error creating TGR placeholders: {e}")
        return 0


def show_tgr_stats(db_path: str = "greyhound_racing_data.db"):
    """Show TGR coverage statistics."""

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Total unique dogs
        cursor.execute(
            "SELECT COUNT(DISTINCT dog_clean_name) FROM dog_race_data WHERE dog_clean_name IS NOT NULL"
        )
        total_dogs = cursor.fetchone()[0]

        # Dogs with TGR data
        cursor.execute("SELECT COUNT(*) FROM tgr_dog_performance_summary")
        tgr_dogs = cursor.fetchone()[0]

        # Placeholder vs real data
        cursor.execute(
            "SELECT COUNT(*) FROM tgr_dog_performance_summary WHERE performance_data LIKE '%placeholder%'"
        )
        placeholder_count = cursor.fetchone()[0]

        real_tgr = tgr_dogs - placeholder_count

        conn.close()

        coverage = (tgr_dogs / total_dogs * 100) if total_dogs > 0 else 0

        logger.info(f"📊 TGR Coverage Statistics:")
        logger.info(f"  Total unique dogs: {total_dogs:,}")
        logger.info(f"  Dogs with TGR records: {tgr_dogs:,} ({coverage:.1f}%)")
        logger.info(f"  Real TGR data: {real_tgr}")
        logger.info(f"  Placeholder data: {placeholder_count}")

        return {
            "total_dogs": total_dogs,
            "tgr_dogs": tgr_dogs,
            "coverage_percent": coverage,
            "real_tgr": real_tgr,
            "placeholders": placeholder_count,
        }

    except Exception as e:
        logger.error(f"Error getting TGR stats: {e}")
        return {}


def main():
    """Main function."""

    logger.info("🚀 Creating TGR placeholder records for all dogs...")

    # Show current stats
    show_tgr_stats()

    # Create placeholders
    created = create_tgr_placeholders()

    if created > 0:
        logger.info(f"✅ Successfully created {created} placeholder TGR records")
        logger.info("💡 This will enable:")
        logger.info("  - ML training with TGR feature compatibility")
        logger.info("  - Faster race processing (no TGR lookup delays)")
        logger.info("  - Better prediction pipeline performance")
        logger.info("  - Background TGR data collection can update placeholders")
    else:
        logger.info("ℹ️ No placeholder records needed or creation failed")


if __name__ == "__main__":
    main()
