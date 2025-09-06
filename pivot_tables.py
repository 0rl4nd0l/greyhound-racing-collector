import pandas as pd

# Load your CSV data
# Assuming we load the DataFrame from 'processed/step6_cleanup/Race 1 - BEN - 02 July 2025_cleaned.csv'
data = pd.read_csv("processed/step6_cleanup/Race 1 - BEN - 02 July 2025_cleaned.csv")

# Average race time by track vs. month
pivot_avg_time = pd.pivot_table(
    data, values="TIME", index="TRACK", columns="month", aggfunc="mean"
)

# Count of races by day_of_week vs. track
pivot_race_count = pd.pivot_table(
    data, values="Dog Name", index="day_of_week", columns="TRACK", aggfunc="count"
)

# Assuming 'log_data' is a column indicating presence (1) or absence (0) of log data
# Calculate percentage of missing log data by track vs. quarter
# Here we assume "log_data" presence might be interpreted from available data fields if not explicitly given
# We'll create a mock 'log_data_present' for demonstration
# In reality, you should use the actual log presence indicator from your dataset
data["log_data_present"] = data["Dog Name"].notna().astype(int)  # Mock assumption
pivot_log_data = (
    data.pivot_table(
        values="log_data_present",
        index="TRACK",
        columns=pd.to_datetime(data["DATE"]).dt.to_period("Q"),
        aggfunc=lambda x: 1
        - x.mean(),  # Percentage of missing data is 1 - percentage of present data
    )
    * 100
)

# Print the pivot tables
print("Average race time by track vs. month", pivot_avg_time, sep="\n")
print("Count of races by day_of_week vs. track", pivot_race_count, sep="\n")
print(
    "Percentage of races with missing log data by track vs. quarter",
    pivot_log_data,
    sep="\n",
)
