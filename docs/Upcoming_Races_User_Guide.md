# Upcoming Races User Guide

This guide walks end users through the manual flow for generating predictions from upcoming race CSVs and seeing them immediately in the UI.

Audience: operators, analysts, and developers running the local UI.

Prerequisites
- Application is running (python app.py or equivalent service)
- UPCOMING_RACES_DIR is set (default: ./upcoming_races or ./upcoming_races_temp)
- You have at least one valid race CSV named like: "Race 1 - GOSF - 2025-07-31.csv"

Step 1 — Prepare your CSV
- Ensure required columns exist: race_date, venue_code, race_number, dog_name, box
- Do not include post-race fields like PLC, finish_position, winner, margin
- Save the file using the naming pattern: Race {number} - {VENUE_CODE} - YYYY-MM-DD.csv

[SCREENSHOT 1 PLACEHOLDER]
- Suggested image: A Finder/Explorer window showing the UPCOMING_RACES_DIR and a correctly named CSV file

Step 2 — Place the CSV into UPCOMING_RACES_DIR
- Move or copy your CSV into the configured UPCOMING_RACES_DIR
- If a watcher is configured (DOWNLOADS_WATCH_DIR), it may move files automatically

[SCREENSHOT 2 PLACEHOLDER]
- Suggested image: Terminal or file manager highlighting the UPCOMING_RACES_DIR path

Step 3 — Open the UI and navigate to "Upcoming Races"
- The UI lists CSV files directly from UPCOMING_RACES_DIR
- Your newly added file should appear immediately (no database insert required)

[SCREENSHOT 3 PLACEHOLDER]
- Suggested image: The Upcoming Races page with the new file visible in the list

Step 4 — Select a race to generate predictions
- Click the race in the UI, or call the API /api/predict_single_race_enhanced with the race_filename
- The backend runs PredictionPipelineV4 (with fallbacks) and returns predictions

[SCREENSHOT 4 PLACEHOLDER]
- Suggested image: Predictions view showing ranked results and probabilities

Step 5 — Verify outputs and logs
- Predictions are written to PREDICTIONS_DIR and displayed in the UI
- Check logs/ for any warnings (calibration, missing fields, drift)

Tips
- Use consistent naming so races group by date and venue
- Archive outdated CSVs to archive/upcoming_races/YYYY/MM to keep the list clean
- If a CSV doesn’t appear in the UI, review permissions and paths (see README Troubleshooting)

FAQ
- Q: My file is not visible.
  - A: Confirm UPCOMING_RACES_DIR, filename, permissions, and ensure the UI is pointed at the same directory.
- Q: The API says file not found.
  - A: Make sure the filename matches exactly (including spaces) and the process has access to the folder.
- Q: Can I bulk-predict?
  - A: Use /api/predict_all_upcoming_races_enhanced to process all CSVs in the directory.

