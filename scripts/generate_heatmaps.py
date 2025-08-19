import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

print("Generating Seaborn heatmaps for pivot tables...")

# Load data from all cleaned CSV files in step6_cleanup directory
data_files = glob.glob('processed/step6_cleanup/*_cleaned.csv')
if not data_files:
    # Fallback to specific file if pattern doesn't match
    data_files = ['processed/step6_cleanup/Race 1 - BEN - 02 July 2025_cleaned.csv']

print(f"Found {len(data_files)} data files to process")

# Load and combine all data
all_data = []
for file in data_files:
    if os.path.exists(file):
        try:
            df = pd.read_csv(file) 
            all_data.append(df)
            print(f"Loaded: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            
if not all_data:
    print("No data files could be loaded!")
    exit(1)

# Combine all dataframes
data = pd.concat(all_data, ignore_index=True)
print(f"Combined dataset shape: {data.shape}")

# Create pivot tables (same as in pivot_tables.py)
print("Creating pivot tables...")

# 1. Average race time by track vs. month
try:
    pivot_avg_time = pd.pivot_table(data, values='TIME', index='TRACK', columns='month', aggfunc='mean')
    print(f"Pivot 1 shape: {pivot_avg_time.shape}")
except Exception as e:
    print(f"Error creating avg_time pivot: {e}")
    pivot_avg_time = None

# 2. Count of races by day_of_week vs. track  
try:
    pivot_race_count = pd.pivot_table(data, values='Dog Name', index='day_of_week', columns='TRACK', aggfunc='count')
    print(f"Pivot 2 shape: {pivot_race_count.shape}")
except Exception as e:
    print(f"Error creating race_count pivot: {e}")
    pivot_race_count = None

# 3. Percentage of missing log data by track vs. quarter
try:
    data['log_data_present'] = data['Dog Name'].notna().astype(int)  # Mock assumption
    # Handle date parsing more robustly
    data['DATE_clean'] = pd.to_datetime(data['DATE'], errors='coerce')
    pivot_log_data = data.pivot_table(
        values='log_data_present',
        index='TRACK',
        columns=data['DATE_clean'].dt.to_period('Q'),
        aggfunc=lambda x: 1 - x.mean()  # Percentage of missing data
    ) * 100
    print(f"Pivot 3 shape: {pivot_log_data.shape}")
except Exception as e:
    print(f"Error creating log_data pivot: {e}")
    pivot_log_data = None

# Create heatmaps
print("Generating heatmap visualizations...")

# Heatmap 1: Average Race Time by Track vs. Month
if pivot_avg_time is not None and not pivot_avg_time.empty:
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_avg_time, annot=True, fmt=".1f", cmap="viridis")
    plt.title('Average Race Time by Track vs. Month')
    plt.tight_layout()
    plt.savefig('reports/figures/avg_race_time_by_track_vs_month.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: avg_race_time_by_track_vs_month.png")
else:
    print("✗ Skipped: avg_race_time_by_track_vs_month.png (no valid data)")

# Heatmap 2: Race Count by Day of Week vs. Track
if pivot_race_count is not None and not pivot_race_count.empty:
    plt.figure(figsize=(12, 8)) 
    sns.heatmap(pivot_race_count, annot=True, fmt=".0f", cmap="viridis")
    plt.title('Race Count by Day of Week vs. Track')
    plt.tight_layout()
    plt.savefig('reports/figures/race_count_by_day_vs_track.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: race_count_by_day_vs_track.png")
else:
    print("✗ Skipped: race_count_by_day_vs_track.png (no valid data)")

# Heatmap 3: Missing Log Data by Track vs. Quarter
if pivot_log_data is not None and not pivot_log_data.empty:
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_log_data, annot=True, fmt=".1f", cmap="viridis")
    plt.title('Missing Log Data by Track vs. Quarter (%)')
    plt.tight_layout()
    plt.savefig('reports/figures/missing_log_data_by_track_vs_quarter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: missing_log_data_by_track_vs_quarter.png")
else:
    print("✗ Skipped: missing_log_data_by_track_vs_quarter.png (no valid data)")

print("\nHeatmap generation complete!")
print(f"Output directory: {os.path.abspath('reports/figures')}")

