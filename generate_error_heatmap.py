import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np
from collections import defaultdict

# File paths
AUDIT_DIR = "audit"
ERROR_TYPES_FILE = os.path.join(AUDIT_DIR, "error_types_summary.json")
VALIDATION_SUMMARY = os.path.join(AUDIT_DIR, "validation_summary.parquet")
HEATMAP_OUTPUT = os.path.join(AUDIT_DIR, "error_types_heatmap.png")

def generate_error_heatmap():
    """Generate a heatmap visualization of error types to guide fixes"""
    
    # Load error types data
    if not os.path.exists(ERROR_TYPES_FILE):
        print(f"Error types file not found: {ERROR_TYPES_FILE}")
        return
        
    with open(ERROR_TYPES_FILE, 'r') as f:
        error_type_counts = json.load(f)
    
    # Load validation summary for more detailed analysis
    if os.path.exists(VALIDATION_SUMMARY):
        df_summary = pd.read_parquet(VALIDATION_SUMMARY)
        
        # Create a matrix for directory vs error type analysis
        directory_error_matrix = defaultdict(lambda: defaultdict(int))
        
        for _, row in df_summary.iterrows():
            file_path = row['file_path']
            directory = os.path.dirname(file_path)
            # Simplify directory names
            if 'processed' in directory:
                dir_name = 'processed'
            elif 'unprocessed' in directory:
                dir_name = 'unprocessed'
            elif 'form_guides' in directory:
                dir_name = 'form_guides'
            else:
                dir_name = 'other'
            
            for error_type in row.get('error_types', []):
                directory_error_matrix[dir_name][error_type] += 1
        
        # Convert to DataFrame for heatmap
        if directory_error_matrix:
            # Create matrix DataFrame
            all_error_types = set()
            for dir_errors in directory_error_matrix.values():
                all_error_types.update(dir_errors.keys())
            
            all_dirs = list(directory_error_matrix.keys())
            all_error_types = list(all_error_types)
            
            matrix_data = []
            for dir_name in all_dirs:
                row_data = []
                for error_type in all_error_types:
                    row_data.append(directory_error_matrix[dir_name][error_type])
                matrix_data.append(row_data)
            
            heatmap_df = pd.DataFrame(matrix_data, index=all_dirs, columns=all_error_types)
            
            # Create the heatmap
            plt.figure(figsize=(14, 8))
            
            # Create two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            
            # Top plot: Overall error type frequency
            error_types_sorted = sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            error_names = [item[0] for item in error_types_sorted]
            error_counts = [item[1] for item in error_types_sorted]
            
            bars = ax1.bar(range(len(error_names)), error_counts, color='lightcoral')
            ax1.set_xlabel('Error Types')
            ax1.set_ylabel('Number of Files')
            ax1.set_title('Most Common Error Types Across All Files')
            ax1.set_xticks(range(len(error_names)))
            ax1.set_xticklabels(error_names, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, error_counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom')
            
            # Bottom plot: Directory vs Error Type heatmap
            if not heatmap_df.empty:
                sns.heatmap(heatmap_df, annot=True, fmt='d', cmap='Reds', 
                           ax=ax2, cbar_kws={'label': 'Number of Files'})
                ax2.set_title('Error Types by Directory')
                ax2.set_xlabel('Error Types')
                ax2.set_ylabel('Directories')
            else:
                ax2.text(0.5, 0.5, 'No directory-specific data available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Error Types by Directory - No Data')
            
            plt.tight_layout()
            plt.savefig(HEATMAP_OUTPUT, dpi=300, bbox_inches='tight')
            print(f"Error heatmap saved to: {HEATMAP_OUTPUT}")
            
            # Generate recommendations
            recommendations = generate_fix_recommendations(error_type_counts, df_summary if 'df_summary' in locals() else None)
            
            # Save recommendations
            recommendations_file = os.path.join(AUDIT_DIR, "fix_recommendations.json")
            with open(recommendations_file, 'w') as f:
                json.dump(recommendations, f, indent=2)
            
            print(f"Fix recommendations saved to: {recommendations_file}")
            
        else:
            print("No directory-error data available for heatmap")
    else:
        print(f"Validation summary not found: {VALIDATION_SUMMARY}")

def generate_fix_recommendations(error_type_counts, df_summary=None):
    """Generate actionable recommendations based on error analysis"""
    
    recommendations = {
        "priority_fixes": [],
        "systematic_improvements": [],
        "data_quality_actions": []
    }
    
    # Priority fixes based on most common errors
    sorted_errors = sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)
    
    for error_type, count in sorted_errors[:5]:  # Top 5 errors
        if error_type == "missing_columns":
            recommendations["priority_fixes"].append({
                "error": error_type,
                "count": count,
                "action": "Standardize CSV headers across all form guide files",
                "impact": "High - affects file parseability",
                "effort": "Medium"
            })
        elif error_type == "high_null_percentage":
            recommendations["priority_fixes"].append({
                "error": error_type,
                "count": count,
                "action": "Implement data imputation strategies for missing values",
                "impact": "High - affects data quality",
                "effort": "High"
            })
        elif error_type == "read_error":
            recommendations["priority_fixes"].append({
                "error": error_type,
                "count": count,
                "action": "Fix file encoding and format issues",
                "impact": "Critical - files cannot be processed",
                "effort": "Medium"
            })
        elif error_type == "invalid_weight_format":
            recommendations["data_quality_actions"].append({
                "error": error_type,
                "count": count,
                "action": "Validate and standardize weight data format",
                "impact": "Medium - affects ML model training",
                "effort": "Low"
            })
        elif error_type == "extra_columns":
            recommendations["systematic_improvements"].append({
                "error": error_type,
                "count": count,
                "action": "Create column mapping and filtering system",
                "impact": "Low - informational",
                "effort": "Low"
            })
    
    # Calculate overall statistics
    if df_summary is not None:
        total_files = len(df_summary)
        failed_files = len(df_summary[df_summary['success'] == False])
        avg_quarantine_rate = df_summary['pct_rows_quarantined'].mean()
        
        recommendations["overall_health"] = {
            "total_files_analyzed": total_files,
            "files_with_issues": failed_files,
            "failure_rate": failed_files / total_files if total_files > 0 else 0,
            "average_quarantine_rate": avg_quarantine_rate,
            "health_score": max(0, 100 - (failed_files / total_files * 100) - (avg_quarantine_rate * 100)) if total_files > 0 else 0
        }
    
    return recommendations

if __name__ == "__main__":
    generate_error_heatmap()
