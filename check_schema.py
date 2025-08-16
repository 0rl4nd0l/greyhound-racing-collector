import pandas as pd
from sqlite3 import connect

# Connect to the database
conn = connect('/Users/orlandolee/greyhound_racing_collector/databases/race_data.db')

# Load 5 random entries from the dog_performances table
query = "SELECT * FROM dog_performances ORDER BY RANDOM() LIMIT 5;"
df = pd.read_sql(query, conn)

# Coerce types based on database schema
expected_types = {
    'performance_id': 'Int64',
    'race_id': 'Int64',
    'dog_name': 'str',
    'box_number': 'Int64',
    'finish_position': 'Float64',
    'race_time': 'Float64',
    'weight': 'Float64',
    'trainer': 'str',
    'odds': 'str',
    'margin': 'str',
    'sectional_time': 'str',
    'split_times': 'str'
}

df = df.astype(expected_types)

# Check for NOT NULL violations
not_null_violations = df[['race_id', 'dog_name', 'box_number']].isnull().sum()

# Check for numeric range violations
weight_violations = df[~df['weight'].between(20, 40)].shape[0]
box_number_violations = df[~df['box_number'].between(1, 8)].shape[0]

# Write results to schema_conformance_report.md
with open('/Users/orlandolee/greyhound_racing_collector/audit/schema_conformance_report.md', 'a') as f:
    f.write('### NOT NULL Violations\n')
    f.write(not_null_violations.to_string())
    f.write('\n\n')
    f.write('### Numeric Range Violations\n')
    f.write(f'Weight violations: {weight_violations}\n')
    f.write(f'Box number violations: {box_number_violations}\n')

# Close database connection
conn.close()
