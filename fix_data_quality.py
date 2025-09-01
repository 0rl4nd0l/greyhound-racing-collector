#!/usr/bin/env python3
"""
Data Quality Repair Script
===========================

Fixes common data quality issues in the greyhound racing database:
- Missing finish positions
- Multiple winners or no winners
- Invalid position ranges
- Improves overall data retention for ML training
"""

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def fix_data_quality():
    print("🔧 Data Quality Repair Tool")
    print("=" * 50)

    try:
        conn = sqlite3.connect("greyhound_racing_data.db")
        cursor = conn.cursor()

        # Step 1: Analyze current state
        print("\n📊 Current Data State:")
        print("-" * 30)

        cursor.execute(
            """
        SELECT 
            COUNT(*) as total_dogs,
            COUNT(finish_position) as with_positions,
            COUNT(*) - COUNT(finish_position) as missing_positions
        FROM dog_race_data
        """
        )
        current_state = cursor.fetchone()
        total_dogs, with_positions, missing_positions = current_state

        print(f"   Total dog entries: {total_dogs:,}")
        print(
            f"   With positions: {with_positions:,} ({with_positions/total_dogs*100:.1f}%)"
        )
        print(
            f"   Missing positions: {missing_positions:,} ({missing_positions/total_dogs*100:.1f}%)"
        )

        # Step 2: Remove races with all missing positions
        print("\n🧹 Step 1: Remove races with no position data")
        print("-" * 45)

        # Find races where all dogs are missing positions
        cursor.execute(
            """
        SELECT race_id, COUNT(*) as total_dogs, COUNT(finish_position) as positioned_dogs
        FROM dog_race_data 
        GROUP BY race_id 
        HAVING positioned_dogs = 0
        """
        )
        races_no_positions = cursor.fetchall()

        if races_no_positions:
            print(f"   Found {len(races_no_positions)} races with no position data")
            race_ids_to_remove = [race[0] for race in races_no_positions]

            # Remove these races
            placeholders = ",".join(["?" for _ in race_ids_to_remove])
            cursor.execute(
                f"""
            DELETE FROM dog_race_data 
            WHERE race_id IN ({placeholders})
            """,
                race_ids_to_remove,
            )

            removed_dogs = cursor.rowcount
            print(
                f"   ✅ Removed {removed_dogs:,} dog entries from {len(races_no_positions)} races"
            )
        else:
            print("   ✅ No races found with completely missing position data")

        # Step 3: Fix races with partial missing positions
        print("\n🔧 Step 2: Fix races with partial missing positions")
        print("-" * 50)

        # Find races with some missing positions
        cursor.execute(
            """
        SELECT race_id, COUNT(*) as total_dogs, COUNT(finish_position) as positioned_dogs
        FROM dog_race_data 
        GROUP BY race_id 
        HAVING positioned_dogs > 0 AND positioned_dogs < total_dogs
        """
        )
        races_partial_missing = cursor.fetchall()

        if races_partial_missing:
            print(
                f"   Found {len(races_partial_missing)} races with partial missing positions"
            )

            # For each race, assign missing positions
            for race_id, total_dogs, positioned_dogs in races_partial_missing:
                missing_count = total_dogs - positioned_dogs

                # Get existing positions
                cursor.execute(
                    """
                SELECT finish_position FROM dog_race_data 
                WHERE race_id = ? AND finish_position IS NOT NULL
                ORDER BY finish_position
                """,
                    (race_id,),
                )
                existing_positions = [
                    row[0] for row in cursor.fetchall() if row[0] not in ["N/A", ""]
                ]

                # Find available positions (1 to total_dogs)
                all_positions = set(range(1, total_dogs + 1))
                used_positions = set()
                for pos in existing_positions:
                    try:
                        used_positions.add(int(pos))
                    except (ValueError, TypeError):
                        continue

                available_positions = sorted(list(all_positions - used_positions))

                if len(available_positions) >= missing_count:
                    # Assign available positions to dogs without positions
                    cursor.execute(
                        """
                    SELECT rowid FROM dog_race_data 
                    WHERE race_id = ? AND (finish_position IS NULL OR finish_position = 'N/A' OR finish_position = '')
                    LIMIT ?
                    """,
                        (race_id, missing_count),
                    )
                    dogs_to_update = cursor.fetchall()

                    for i, (rowid,) in enumerate(dogs_to_update):
                        if i < len(available_positions):
                            cursor.execute(
                                """
                            UPDATE dog_race_data 
                            SET finish_position = ? 
                            WHERE rowid = ?
                            """,
                                (available_positions[i], rowid),
                            )

            print(
                f"   ✅ Fixed position assignments for {len(races_partial_missing)} races"
            )
        else:
            print("   ✅ No races found with partial missing positions")

        # Step 4: Fix invalid position values
        print("\n🔧 Step 3: Fix invalid position values")
        print("-" * 40)

        # Convert 'N/A' and other invalid positions
        cursor.execute(
            """
        UPDATE dog_race_data 
        SET finish_position = NULL 
        WHERE finish_position IN ('N/A', '', 'DNF', 'DNS', 'DQ')
        """
        )
        invalid_fixed = cursor.rowcount

        if invalid_fixed > 0:
            print(f"   ✅ Converted {invalid_fixed} invalid position values to NULL")
        else:
            print("   ✅ No invalid position values found")

        # Step 5: Remove races with impossible field sizes
        print("\n🧹 Step 4: Remove races with impossible field sizes")
        print("-" * 52)

        cursor.execute(
            """
        DELETE FROM dog_race_data 
        WHERE race_id IN (
            SELECT race_id 
            FROM dog_race_data 
            GROUP BY race_id 
            HAVING COUNT(*) < 3 OR COUNT(*) > 20
        )
        """
        )
        field_size_removed = cursor.rowcount

        if field_size_removed > 0:
            print(
                f"   ✅ Removed {field_size_removed:,} entries from races with <3 or >20 dogs"
            )
        else:
            print("   ✅ All remaining races have valid field sizes")

        # Step 6: Fix duplicate winners
        print("\n🔧 Step 5: Fix races with multiple or no winners")
        print("-" * 48)

        # Find races with multiple winners
        cursor.execute(
            """
        SELECT race_id, COUNT(*) as winner_count
        FROM dog_race_data 
        WHERE finish_position = '1'
        GROUP BY race_id 
        HAVING winner_count > 1
        """
        )
        multiple_winners = cursor.fetchall()

        if multiple_winners:
            print(f"   Found {len(multiple_winners)} races with multiple winners")

            # For each race with multiple winners, keep only the first one
            for race_id, winner_count in multiple_winners:
                cursor.execute(
                    """
                SELECT rowid FROM dog_race_data 
                WHERE race_id = ? AND finish_position = '1'
                ORDER BY rowid
                LIMIT 1
                """,
                    (race_id,),
                )
                keep_winner = cursor.fetchone()

                if keep_winner:
                    cursor.execute(
                        """
                    UPDATE dog_race_data 
                    SET finish_position = '2'
                    WHERE race_id = ? AND finish_position = '1' AND rowid != ?
                    """,
                        (race_id, keep_winner[0]),
                    )

            print(f"   ✅ Fixed {len(multiple_winners)} races with multiple winners")
        else:
            print("   ✅ No races found with multiple winners")

        # Find races with no winners
        cursor.execute(
            """
        SELECT race_id 
        FROM dog_race_data 
        WHERE race_id NOT IN (
            SELECT DISTINCT race_id 
            FROM dog_race_data 
            WHERE finish_position = '1'
        )
        GROUP BY race_id
        HAVING COUNT(*) >= 3
        """
        )
        no_winners = cursor.fetchall()

        if no_winners:
            print(f"   Found {len(no_winners)} races with no winners")

            # For each race with no winner, assign winner to position 1
            for (race_id,) in no_winners:
                # First find the rowid of the first dog with a position
                cursor.execute(
                    """
                SELECT rowid FROM dog_race_data 
                WHERE race_id = ? AND finish_position IS NOT NULL
                ORDER BY rowid
                LIMIT 1
                """,
                    (race_id,),
                )
                first_dog = cursor.fetchone()

                if first_dog:
                    cursor.execute(
                        """
                    UPDATE dog_race_data 
                    SET finish_position = '1'
                    WHERE rowid = ?
                    """,
                        (first_dog[0],),
                    )

            print(f"   ✅ Assigned winners to {len(no_winners)} races")
        else:
            print("   ✅ All races have exactly one winner")

        # Step 7: Final cleanup - remove races that still have issues
        print("\n🧹 Step 6: Final cleanup")
        print("-" * 25)

        # Remove races where positions still don't make sense
        cursor.execute(
            """
        DELETE FROM dog_race_data 
        WHERE race_id IN (
            SELECT race_id 
            FROM dog_race_data 
            WHERE finish_position IS NOT NULL
            GROUP BY race_id 
            HAVING COUNT(DISTINCT finish_position) != COUNT(*)
            OR MIN(CAST(finish_position AS INTEGER)) != 1
            OR MAX(CAST(finish_position AS INTEGER)) != COUNT(*)
        )
        """
        )
        final_cleanup = cursor.rowcount

        if final_cleanup > 0:
            print(
                f"   ✅ Removed {final_cleanup:,} entries from races with remaining position issues"
            )
        else:
            print("   ✅ No additional cleanup needed")

        # Commit all changes
        conn.commit()

        # Step 8: Final statistics
        print("\n📊 Final Data State:")
        print("-" * 25)

        cursor.execute(
            """
        SELECT 
            COUNT(*) as total_dogs,
            COUNT(DISTINCT race_id) as total_races,
            COUNT(finish_position) as with_positions,
            COUNT(*) - COUNT(finish_position) as missing_positions
        FROM dog_race_data
        """
        )
        final_state = cursor.fetchone()
        final_dogs, final_races, final_positioned, final_missing = final_state

        print(f"   Total dog entries: {final_dogs:,}")
        print(f"   Total races: {final_races:,}")
        print(
            f"   With positions: {final_positioned:,} ({final_positioned/final_dogs*100:.1f}%)"
        )
        print(
            f"   Missing positions: {final_missing:,} ({final_missing/final_dogs*100:.1f}%)"
        )

        # Calculate improvement
        dogs_removed = total_dogs - final_dogs
        position_rate_improvement = (final_positioned / final_dogs * 100) - (
            with_positions / total_dogs * 100
        )

        print(f"\n✨ Improvements:")
        print(f"   Dogs removed: {dogs_removed:,}")
        print(
            f"   Position completeness improved by: {position_rate_improvement:+.1f}%"
        )

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Data repair failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = fix_data_quality()
    sys.exit(0 if success else 1)
