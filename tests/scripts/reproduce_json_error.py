#!/usr/bin/env python3
"""
Minimal script to reproduce JSONDecodeError from EnhancedLogger.load_web_logs()
=====================================================================

This script is designed to reproduce the exact JSONDecodeError by:
1. Importing the EnhancedLogger class
2. Calling the load_web_logs() method 
3. Capturing the full traceback for analysis

Author: AI Assistant
Date: August 3, 2025
"""

import sys
import traceback
from pathlib import Path

# Add the current directory to Python path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main function to reproduce the JSONDecodeError"""
    
    print("=" * 60)
    print("🔍 REPRODUCING JSONDecodeError FROM EnhancedLogger.load_web_logs()")
    print("=" * 60)
    
    try:
        # Import the EnhancedLogger class
        print("📦 Importing EnhancedLogger...")
        from logger import EnhancedLogger
        print("✅ Successfully imported EnhancedLogger")
        
        # Create an instance of EnhancedLogger
        print("🏗️  Creating EnhancedLogger instance...")
        logger = EnhancedLogger(log_dir="./logs")
        print("✅ EnhancedLogger instance created successfully")
        
        # Directly call load_web_logs() method 
        print("📂 Calling load_web_logs() method...")
        logger.load_web_logs()
        print("✅ load_web_logs() completed successfully")
        
    except Exception as e:
        print("\n" + "="*60)
        print("🚨 ERROR CAPTURED!")
        print("="*60)
        
        # Capture the exact error details
        error_type = type(e).__name__
        error_message = str(e)
        
        print(f"Error Type: {error_type}")
        print(f"Error Message: {error_message}")
        print(f"\nFull Exception Details:")
        print(f"Exception Class: {e.__class__.__module__}.{e.__class__.__name__}")
        
        # Get the full traceback
        print("\n" + "-"*60)
        print("FULL TRACEBACK:")
        print("-"*60)
        
        # Print the full traceback to get file paths and line numbers
        traceback.print_exc()
        
        # Also capture traceback as string for storage
        tb_string = traceback.format_exc()
        
        # Store evidence for regression testing
        evidence_file = Path("json_decode_error_evidence.txt")
        with open(evidence_file, "w") as f:
            f.write("JSONDecodeError Evidence - Captured on August 3, 2025\n")
            f.write("="*60 + "\n\n")
            f.write(f"Error Type: {error_type}\n")
            f.write(f"Error Message: {error_message}\n")
            f.write(f"Exception Class: {e.__class__.__module__}.{e.__class__.__name__}\n")
            f.write("\nFull Traceback:\n")
            f.write("-"*40 + "\n")
            f.write(tb_string)
            f.write("\nAdditional Context:\n")
            f.write("-"*40 + "\n")
            f.write("- Called from: reproduce_json_error.py\n")
            f.write("- Method: EnhancedLogger.load_web_logs()\n")
            f.write("- Log directory: ./logs\n")
            f.write(f"- Working directory: {Path.cwd()}\n")
        
        print(f"\n📄 Evidence stored in: {evidence_file.absolute()}")
        
        # Return the error for further analysis
        return e
        
    print("\n✅ No JSONDecodeError encountered - script completed successfully")
    return None

if __name__ == "__main__":
    """
    Run the reproduction script and capture any JSONDecodeError
    """
    print("🎯 Starting JSONDecodeError reproduction script...")
    error = main()
    
    if error:
        print(f"\n🎯 ERROR REPRODUCTION SUCCESSFUL!")
        print(f"   Error Type: {type(error).__name__}")
        print(f"   Error Message: {str(error)}")
        sys.exit(1)
    else:
        print(f"\n✅ No error found - EnhancedLogger.load_web_logs() works correctly")
        sys.exit(0)
