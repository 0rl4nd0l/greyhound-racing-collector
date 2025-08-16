#!/usr/bin/env python3
import os
import shutil
import time

# Define the log directory and archive directory
log_directory = './logs'
archive_directory = './backups/archives'

# Create the archive directory if it doesn't exist
os.makedirs(archive_directory, exist_ok=True)

# Define the age in seconds for old files (30 days)
age_of_files = 30 * 24 * 60 * 60

# Function to archive old log files
def archive_old_logs():
    current_time = time.time()

    # Walk through all files in the log directory
    for root, dirs, files in os.walk(log_directory):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if the file is old
            if os.stat(file_path).st_mtime < (current_time - age_of_files):
                # Compress the old file
                shutil.make_archive(file_path, 'zip', root, file)
                # Move the compressed file to the archive directory
                shutil.move(f'{file_path}.zip', archive_directory)
                # Remove the original file
                os.remove(file_path)

if __name__ == '__main__':
    archive_old_logs()
    print('Old logs have been archived successfully.')

