import sqlite3


def upgrade():
    """Add start_datetime to race_metadata table."""
    conn = sqlite3.connect("greyhound_racing_data.db")
    cursor = conn.cursor()

    try:
        # Check if the column already exists
        cursor.execute("PRAGMA table_info(race_metadata)")
        columns = [column[1] for column in cursor.fetchall()]

        if "start_datetime" not in columns:
            cursor.execute(
                "ALTER TABLE race_metadata ADD COLUMN start_datetime DATETIME"
            )
            print("Column 'start_datetime' added to 'race_metadata' table.")
        else:
            print("Column 'start_datetime' already exists in 'race_metadata' table.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")

    finally:
        conn.commit()
        conn.close()


if __name__ == "__main__":
    upgrade()
