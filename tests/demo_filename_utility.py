#!/usr/bin/env python3
"""
Demonstration script for the new filename utility functions.

This script shows how the get_filename_for_race_id function can be used
to replace duplicated filename search logic across endpoints.
"""

from utils.file_naming import get_filename_for_race_id, get_race_id_from_filename

def demo_usage():
    """Demonstrate usage of the filename utility functions."""
    
    print("🎯 Filename Utility Function Demonstration")
    print("=" * 50)
    
    # Example 1: Search for a race file using default paths
    print("\n1. Searching with default paths (./upcoming_races, ./historical_races)")
    race_id = "Race_01_MAN_2025-01-15"
    filename, full_path = get_filename_for_race_id(race_id)
    
    if filename:
        print(f"   ✅ Found: {filename}")
        print(f"   📁 Path: {full_path}")
    else:
        print(f"   ❌ File not found for race_id: {race_id}")
    
    # Example 2: Search with custom paths
    print("\n2. Searching with custom paths")
    custom_paths = ["./custom_dir1", "./custom_dir2", "./archive"]
    filename, full_path = get_filename_for_race_id(race_id, custom_paths)
    
    if filename:
        print(f"   ✅ Found: {filename}")
        print(f"   📁 Path: {full_path}")
    else:
        print(f"   ❌ File not found in custom paths for race_id: {race_id}")
    
    # Example 3: Extract race_id from filename
    print("\n3. Extracting race_id from filename")
    example_filenames = [
        "Race_01_MAN_2025-01-15.csv",
        "Race 05 GEE 2025-07-30.csv",
        "Race-10-ALB-2025-12-25.csv",
        "/path/to/file/01_MAN_2025-01-15.csv"
    ]
    
    for filename in example_filenames:
        extracted_race_id = get_race_id_from_filename(filename)
        print(f"   📄 {filename}")
        print(f"   🎯 Extracted race_id: {extracted_race_id}")
    
    print("\n" + "=" * 50)
    print("✨ This utility prevents duplication of filename search logic!")
    print("🔄 Use it in endpoints instead of writing custom search code.")


def show_before_after_comparison():
    """Show comparison between old duplicated code and new utility usage."""
    
    print("\n🔄 Before & After Comparison")
    print("=" * 30)
    
    print("\n❌ BEFORE (Duplicated logic in endpoints):")
    print("""
    # Duplicated filename search logic (from app.py)
    possible_filenames = [
        f"{race_id}.csv",
        f"Race {race_id}.csv", 
        f"Race_{race_id}.csv"
    ]
    
    race_file_path = None
    for filename_candidate in possible_filenames:
        candidate_path = os.path.join(UPCOMING_DIR, filename_candidate)
        if os.path.exists(candidate_path):
            race_filename = filename_candidate
            race_file_path = candidate_path
            break
    
    if not race_file_path:
        for filename_candidate in possible_filenames:
            candidate_path = os.path.join(HISTORICAL_DIR, filename_candidate)
            if os.path.exists(candidate_path):
                race_filename = filename_candidate  
                race_file_path = candidate_path
                break
    
    # ... more duplicate search logic for partial matches
    """)
    
    print("\n✅ AFTER (Using centralized utility):")
    print("""
    from utils.file_naming import get_filename_for_race_id
    
    # Clean, centralized filename search
    filename, full_path = get_filename_for_race_id(race_id, [UPCOMING_DIR, HISTORICAL_DIR])
    
    if not full_path:
        return jsonify({
            "success": False,
            "message": f"No race file found for race_id '{race_id}'"
        }), 404
    """)
    
    print("\n🎉 Benefits:")
    print("   • ✨ No more duplicated search logic")
    print("   • 🧪 Well-tested and reliable")
    print("   • 🔧 Easy to maintain and update")
    print("   • 📝 Consistent behavior across endpoints")
    print("   • 🎯 Supports multiple filename patterns")


if __name__ == "__main__":
    demo_usage()
    show_before_after_comparison()
