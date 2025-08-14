import os

import pandas as pd

from enhanced_comprehensive_processor import EnhancedComprehensiveProcessor

# Initialize processor
processor = EnhancedComprehensiveProcessor()

# Directory paths
processed_dir = './processed'
processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]

# Process each file
for file in processed_files:
    file_path = os.path.join(processed_dir, file)
    print(f'Processing {file}...')
    processor.process_csv_file(file_path)
