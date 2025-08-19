"""
Example showing how to integrate parse_race_csv_meta into Flask endpoints.
This demonstrates the usage patterns for the CSV metadata extractor.
"""

import os
from utils.csv_metadata import parse_race_csv_meta

# Example Flask endpoint integration
def create_csv_metadata_endpoint(app):
    """
    Example of how to integrate parse_race_csv_meta into a Flask endpoint.
    This could be added to the main app.py file.
    """
    
    @app.route("/api/csv/metadata", methods=["GET"])
    def api_csv_metadata():
        """
        API endpoint to extract metadata from CSV files.
        
        Query Parameters:
        - file_path: Path to CSV file (required)
        - relative_to: Base directory for relative paths (optional)
        
        Returns:
        JSON response with CSV metadata or error information
        """
        
        try:
            # Get file path from query parameters
            file_path = request.args.get("file_path")
            relative_to = request.args.get("relative_to", "./upcoming_races")
            
            if not file_path:
                return jsonify({
                    "success": False,
                    "error": "file_path parameter is required"
                }), 400
            
            # Handle relative paths
            if not os.path.isabs(file_path):
                file_path = os.path.join(relative_to, file_path)
            
            # Extract metadata using our utility function
            metadata = parse_race_csv_meta(file_path)
            
            # Return success response with metadata
            return jsonify({
                "success": True,
                "metadata": metadata
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }), 500


def analyze_csv_batch(csv_directory: str, pattern: str = "*.csv"):
    """
    Example of batch processing CSV files to extract metadata.
    
    Args:
        csv_directory: Directory containing CSV files
        pattern: Glob pattern for file matching
        
    Returns:
        List of metadata dictionaries for all processed files
    """
    
    import glob
    
    results = []
    csv_files = glob.glob(os.path.join(csv_directory, pattern))
    
    for csv_file in csv_files:
        metadata = parse_race_csv_meta(csv_file)
        results.append(metadata)
    
    return results


def filter_races_by_criteria(csv_directory: str, **criteria):
    """
    Example of filtering races based on extracted metadata.
    
    Args:
        csv_directory: Directory containing CSV files
        **criteria: Filter criteria (venue, distance, grade, etc.)
        
    Returns:
        List of files matching the criteria with their metadata
    """
    
    matching_races = []
    all_metadata = analyze_csv_batch(csv_directory)
    
    for metadata in all_metadata:
        if metadata["status"] != "success":
            continue  # Skip files with errors
            
        # Check if metadata matches all criteria
        match = True
        for key, value in criteria.items():
            if key in metadata:
                if isinstance(value, str):
                    # Case-insensitive string matching
                    if metadata[key].upper() != value.upper():
                        match = False
                        break
                elif metadata[key] != value:
                    match = False
                    break
        
        if match:
            matching_races.append(metadata)
    
    return matching_races


# Example usage functions
def example_usage():
    """
    Examples of how to use the parse_race_csv_meta function.
    """
    
    # Example 1: Single file metadata extraction
    print("=== Example 1: Single File ===")
    sample_file = "./archive/corrupt_or_legacy_race_files/20250730162231_Race 3 - TAR - 28 June 2025.csv"
    metadata = parse_race_csv_meta(sample_file)
    print(f"Race {metadata['race_number']} at {metadata['venue']} on {metadata['race_date']}")
    print(f"Distance: {metadata['distance']}m, Grade: {metadata['grade']}, Field Size: {metadata['field_size']}")
    print(f"Status: {metadata['status']}, Source: {metadata['source']}")
    
    # Example 2: Batch processing (if directory exists)
    csv_dir = "./archive/corrupt_or_legacy_race_files"
    if os.path.exists(csv_dir):
        print(f"\n=== Example 2: Batch Processing ===")
        batch_results = analyze_csv_batch(csv_dir)
        successful_files = [r for r in batch_results if r["status"] == "success"]
        error_files = [r for r in batch_results if r["status"] == "error"]
        
        print(f"Processed {len(batch_results)} files:")
        print(f"- {len(successful_files)} successful")
        print(f"- {len(error_files)} errors")
        
        # Show sample of successful results
        if successful_files:
            print("\nSample successful results:")
            for result in successful_files[:3]:  # Show first 3
                print(f"  {result['filename']}: Race {result['race_number']} at {result['venue']}")
    
    # Example 3: Filtering races
    if os.path.exists(csv_dir):
        print(f"\n=== Example 3: Filtering ===")
        
        # Find all races at TAREE/TAR venue
        taree_races = filter_races_by_criteria(csv_dir, venue="TAR")
        print(f"Found {len(taree_races)} races at TAREE")
        
        # Find all races with distance 300m
        distance_300_races = filter_races_by_criteria(csv_dir, distance="300")
        print(f"Found {len(distance_300_races)} races at 300m distance")


if __name__ == "__main__":
    # Run examples if script is executed directly
    example_usage()
