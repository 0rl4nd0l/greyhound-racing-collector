#!/usr/bin/env python3
"""
File Manager Launcher
====================
Easy access to file management tools.
"""

import sys
import subprocess
import os

def main():
    print("🐕 Greyhound Racing File Manager")
    print("=" * 40)
    print("1. Launch Web UI (Streamlit)")
    print("2. Run Command-Line Inventory")
    print("3. Quick File Count")
    print("4. Exit")
    print("=" * 40)
    
    while True:
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            print("\nLaunching Streamlit UI...")
            print("Note: This will open in your web browser")
            try:
                subprocess.run(["streamlit", "run", "file_manager_ui.py"], check=True)
            except subprocess.CalledProcessError:
                print("Error: Streamlit not found. Install with: pip install streamlit plotly")
            except KeyboardInterrupt:
                print("\nStreamlit UI closed.")
            break
            
        elif choice == "2":
            print("\nRunning detailed inventory scan...")
            try:
                subprocess.run(["python", "file_inventory.py"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running inventory: {e}")
            break
            
        elif choice == "3":
            print("\nQuick file count...")
            try:
                # Quick counts using find commands
                csv_result = subprocess.run([
                    "find", ".", "-name", "*.csv", 
                    "-not", "-path", "./backup_before_cleanup/*",
                    "-not", "-path", "./cleanup_archive/*"
                ], capture_output=True, text=True)
                
                json_result = subprocess.run([
                    "find", ".", "-name", "*.json",
                    "-not", "-path", "./backup_before_cleanup/*", 
                    "-not", "-path", "./cleanup_archive/*"
                ], capture_output=True, text=True)
                
                race_result = subprocess.run([
                    "find", ".", "-name", "Race_*.csv",
                    "-not", "-path", "./backup_before_cleanup/*",
                    "-not", "-path", "./cleanup_archive/*"
                ], capture_output=True, text=True)
                
                csv_count = len([line for line in csv_result.stdout.strip().split('\n') if line])
                json_count = len([line for line in json_result.stdout.strip().split('\n') if line])
                race_count = len([line for line in race_result.stdout.strip().split('\n') if line])
                
                print(f"CSV Files: {csv_count:,}")
                print(f"JSON Files: {json_count:,}")
                print(f"Race CSV Files: {race_count:,}")
                print(f"Total Files: {csv_count + json_count:,}")
                
            except Exception as e:
                print(f"Error getting counts: {e}")
            break
            
        elif choice == "4":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()
