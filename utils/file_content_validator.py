#!/usr/bin/env python3
"""
File Content Validation Utility
===============================

This module provides comprehensive file content validation for the greyhound racing 
collector system. It detects HTML files, empty files, and files that are too small,
logging any skipped files with appropriate warnings.

Key Features:
- Detects HTML content in files that should be CSV
- Checks minimum file size requirements (default: 100 bytes)
- Comprehensive HTML pattern detection
- Detailed file information logging
- Integration with existing file processing workflows

Author: AI Assistant  
Date: January 2025
Version: 1.0.0 - Initial implementation for Step 3
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


class FileContentValidator:
    """
    Validates file content to detect HTML files, empty files, and files below minimum size.
    """
    
    def __init__(self, min_file_size: int = 100, log_skipped_files: bool = True):
        """
        Initialize the file content validator.
        
        Args:
            min_file_size: Minimum file size in bytes (default: 100)
            log_skipped_files: Whether to log skipped file names (default: True)
        """
        self.min_file_size = min_file_size
        self.log_skipped_files = log_skipped_files
        self.skipped_files = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
        # Comprehensive list of HTML indicators
        self.html_indicators = [
            # DOCTYPE declarations
            "<!DOCTYPE",
            "<!doctype",
            
            # HTML tags (case insensitive coverage)
            "<html",
            "<HTML", 
            "<Html",
            
            # Head section tags
            "<head>",
            "<HEAD>",
            "<Head>",
            "</head>",
            "</HEAD>",
            "</Head>",
            
            # Body tags
            "<body>",
            "<BODY>",
            "<Body>",
            "</body>",
            "</BODY>",
            "</Body>",
            
            # Title tags
            "<title>",
            "<TITLE>", 
            "<Title>",
            "</title>",
            "</TITLE>",
            "</Title>",
            
            # Meta tags
            "<meta ",
            "<META ",
            "<Meta ",
            
            # Link tags
            "<link ",
            "<LINK ",
            "<Link ",
            
            # Script tags
            "<script",
            "<SCRIPT",
            "<Script",
            
            # Style tags
            "<style",
            "<STYLE", 
            "<Style",
            
            # Common HTML structural tags
            "<div",
            "<DIV",
            "<Div",
            "<span",
            "<SPAN",
            "<Span",
            "<p>",
            "<P>",
            "<br>",
            "<BR>",
            "<hr>",
            "<HR>",
            
            # Form tags
            "<form",
            "<FORM",
            "<Form",
            "<input",
            "<INPUT",
            "<Input",
            
            # Table tags  
            "<table",
            "<TABLE",
            "<Table",
            "<tr>",
            "<TR>",
            "<Tr>",
            "<td>",
            "<TD>",
            "<Td>",
            
            # Navigation and semantic tags
            "<nav",
            "<NAV",
            "<Nav",
            "<header",
            "<HEADER", 
            "<Header",
            "<footer",
            "<FOOTER",
            "<Footer",
            "<section",
            "<SECTION",
            "<Section",
            "<article",
            "<ARTICLE",
            "<Article",
            
            # List tags
            "<ul>",
            "<UL>",
            "<Ul>",
            "<ol>",
            "<OL>",
            "<Ol>",
            "<li>",
            "<LI>",
            "<Li>",
            
            # Common HTML5 tags
            "<main",
            "<MAIN",
            "<Main",
            "<aside",
            "<ASIDE",
            "<Aside",
            
            # Error page indicators
            "404 Not Found",
            "500 Internal Server Error",
            "Access Denied",
            "Forbidden",
            
            # Server response indicators  
            "HTTP/1.1",
            "HTTP/1.0",
            "Content-Type: text/html",
            "text/html",
            
            # Common web framework indicators
            "<?php",
            "<%",
            "<jsp:",
            "{{",
            "{%"
        ]
        
        # Additional patterns that might indicate non-CSV content
        self.non_csv_patterns = [
            # JSON indicators
            '{"',
            "[{",
            '"success":',
            '"error":',
            '"data":',
            
            # XML indicators  
            "<?xml",
            "<root>",
            "<response>",
            
            # Common error messages
            "Not Found",
            "Page Not Found", 
            "Error 404",
            "Error 500",
            "Internal Server Error",
            "Service Unavailable",
            "Bad Request",
            "Unauthorized",
            "Forbidden",
            "Gateway Timeout",
            
            # Redirect indicators
            "Redirecting",
            "Location:",
            "Moved Permanently",
            "Found. Redirecting to",
            
            # Empty or placeholder responses
            "No data available",
            "Data not found",
            "Empty response",
            "null",
            
            # Server maintenance pages
            "Maintenance Mode",
            "Site Under Construction",
            "Coming Soon",
            "Temporarily Unavailable"
        ]
    
    def validate_file(self, file_path: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate a file for content type and size requirements.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple of (is_valid: bool, message: str, file_info: dict)
        """
        file_info = {
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "file_size": 0,
            "content_type": "unknown",
            "validation_reason": ""
        }
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                message = "File not found"
                file_info["validation_reason"] = message
                return False, message, file_info
            
            # Get file size
            file_size = os.path.getsize(file_path)
            file_info["file_size"] = file_size
            
            # Check minimum file size
            if file_size < self.min_file_size:
                filename = os.path.basename(file_path)
                message = f"File too small: {file_size} bytes (minimum: {self.min_file_size} bytes)"
                file_info["validation_reason"] = message
                file_info["content_type"] = "too_small"
                
                # Log the skipped file
                if self.log_skipped_files:
                    self.logger.warning(f"⚠️  Skipping file '{filename}': Too small ({file_size} bytes < {self.min_file_size} bytes minimum)")
                    self.skipped_files.append({
                        "filename": filename,
                        "reason": "too_small",
                        "file_size": file_size,
                        "path": file_path
                    })
                
                return False, message, file_info
            
            # Check for empty files (size 0)
            if file_size == 0:
                filename = os.path.basename(file_path)
                message = "File is empty (0 bytes)"
                file_info["validation_reason"] = message
                file_info["content_type"] = "empty"
                
                # Log the skipped file
                if self.log_skipped_files:
                    self.logger.warning(f"⚠️  Skipping file '{filename}': File is empty")
                    self.skipped_files.append({
                        "filename": filename,
                        "reason": "empty",
                        "file_size": 0,
                        "path": file_path
                    })
                
                return False, message, file_info
            
            # Read file content to check for HTML patterns
            try:
                # Read first 2KB of file for content analysis (efficient for large files)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content_sample = f.read(2048)  
                    
            except UnicodeDecodeError:
                # Try with different encodings
                encodings = ["latin-1", "cp1252", "ascii"]
                content_sample = ""
                
                for encoding in encodings:
                    try:
                        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
                            content_sample = f.read(2048)
                            break
                    except UnicodeDecodeError:
                        continue
                
                if not content_sample:
                    message = "Unable to read file content - encoding issues"
                    file_info["validation_reason"] = message
                    file_info["content_type"] = "encoding_error"
                    return False, message, file_info
            
            # Check for HTML indicators
            content_lower = content_sample.lower()
            html_found = False
            found_indicators = []
            
            for indicator in self.html_indicators:
                if indicator.lower() in content_lower:
                    html_found = True
                    found_indicators.append(indicator)
            
            # Check for other non-CSV patterns
            non_csv_found = False
            found_patterns = []
            
            for pattern in self.non_csv_patterns:
                if pattern.lower() in content_lower:
                    non_csv_found = True
                    found_patterns.append(pattern)
            
            if html_found or non_csv_found:
                filename = os.path.basename(file_path)
                
                if html_found:
                    message = f"File appears to be HTML, not CSV. Found indicators: {', '.join(found_indicators[:3])}"
                    file_info["content_type"] = "html"
                    file_info["html_indicators"] = found_indicators
                else:
                    message = f"File appears to contain non-CSV content. Found patterns: {', '.join(found_patterns[:3])}"
                    file_info["content_type"] = "non_csv"
                    file_info["non_csv_patterns"] = found_patterns
                
                file_info["validation_reason"] = message
                
                # Log the skipped file
                if self.log_skipped_files:
                    content_type = "HTML" if html_found else "non-CSV"
                    self.logger.warning(f"⚠️  Skipping file '{filename}': Contains {content_type} content, not CSV data")
                    self.skipped_files.append({
                        "filename": filename,
                        "reason": "html_content" if html_found else "non_csv_content",
                        "file_size": file_size,
                        "path": file_path,
                        "indicators": found_indicators if html_found else found_patterns
                    })
                
                return False, message, file_info
            
            # Check if file starts with typical CSV patterns
            csv_patterns = [
                "dog name",
                "dog_name", 
                "box",
                "track",
                "date",
                "place",
                "time",
                "weight",
                "distance",
                '"dog name"',
                '"dog_name"',
                '"box"',
                '"track"',
                '"date"'
            ]
            
            has_csv_pattern = any(pattern in content_lower for pattern in csv_patterns)
            
            if has_csv_pattern:
                file_info["content_type"] = "csv"
                message = f"Valid CSV file ({file_size} bytes)"
                return True, message, file_info
            else:
                # File doesn't contain HTML but also doesn't look like a CSV
                # Still allow it but note the uncertainty
                file_info["content_type"] = "unknown_format"
                message = f"File format uncertain but no HTML detected ({file_size} bytes)"
                return True, message, file_info
                
        except Exception as e:
            message = f"Error validating file: {str(e)}"
            file_info["validation_reason"] = message
            file_info["content_type"] = "validation_error"
            
            if self.log_skipped_files:
                filename = os.path.basename(file_path)
                self.logger.error(f"❌ Error validating file '{filename}': {str(e)}")
                self.skipped_files.append({
                    "filename": filename,
                    "reason": "validation_error",
                    "error": str(e),
                    "path": file_path
                })
            
            return False, message, file_info
    
    def validate_files_batch(self, file_paths: list) -> Dict[str, Any]:
        """
        Validate multiple files in batch.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Dictionary with validation results and statistics
        """
        results = {
            "total_files": len(file_paths),
            "valid_files": [],
            "invalid_files": [],
            "statistics": {
                "valid_count": 0,
                "invalid_count": 0,
                "too_small_count": 0,
                "empty_count": 0,
                "html_count": 0,
                "non_csv_count": 0,
                "error_count": 0,
                "total_valid_size": 0,
                "total_invalid_size": 0
            }
        }
        
        for file_path in file_paths:
            is_valid, message, file_info = self.validate_file(file_path)
            
            if is_valid:
                results["valid_files"].append({
                    "path": file_path,
                    "message": message,
                    "file_info": file_info
                })
                results["statistics"]["valid_count"] += 1
                results["statistics"]["total_valid_size"] += file_info["file_size"]
            else:
                results["invalid_files"].append({
                    "path": file_path,
                    "message": message,
                    "file_info": file_info
                })
                results["statistics"]["invalid_count"] += 1
                results["statistics"]["total_invalid_size"] += file_info["file_size"]
                
                # Count specific error types
                content_type = file_info.get("content_type", "unknown")
                if content_type == "too_small":
                    results["statistics"]["too_small_count"] += 1
                elif content_type == "empty":
                    results["statistics"]["empty_count"] += 1
                elif content_type == "html":
                    results["statistics"]["html_count"] += 1
                elif content_type == "non_csv":
                    results["statistics"]["non_csv_count"] += 1
                else:
                    results["statistics"]["error_count"] += 1
        
        return results
    
    def get_skipped_files_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all skipped files.
        
        Returns:
            Dictionary with skipped files summary
        """
        summary = {
            "total_skipped": len(self.skipped_files),
            "skipped_files": self.skipped_files.copy(),
            "reasons": {}
        }
        
        # Count by reason
        for skipped in self.skipped_files:
            reason = skipped["reason"]
            if reason not in summary["reasons"]:
                summary["reasons"][reason] = 0
            summary["reasons"][reason] += 1
        
        return summary
    
    def clear_skipped_files_log(self):
        """Clear the internal log of skipped files."""
        self.skipped_files.clear()
        
    def print_validation_summary(self, results: Dict[str, Any]):
        """
        Print a formatted summary of validation results.
        
        Args:
            results: Results dictionary from validate_files_batch()
        """
        stats = results["statistics"]
        
        print("\n📊 FILE VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Total files processed: {results['total_files']}")
        print(f"Valid files: {stats['valid_count']} ({stats['valid_count']/results['total_files']*100:.1f}%)")
        print(f"Invalid files: {stats['invalid_count']} ({stats['invalid_count']/results['total_files']*100:.1f}%)")
        
        if stats['invalid_count'] > 0:
            print("\n🔍 INVALID FILE BREAKDOWN:")
            if stats['too_small_count'] > 0:
                print(f"  Files too small: {stats['too_small_count']}")
            if stats['empty_count'] > 0:
                print(f"  Empty files: {stats['empty_count']}")
            if stats['html_count'] > 0:
                print(f"  HTML files: {stats['html_count']}")
            if stats['non_csv_count'] > 0:
                print(f"  Non-CSV content: {stats['non_csv_count']}")
            if stats['error_count'] > 0:
                print(f"  Validation errors: {stats['error_count']}")
        
        # Size statistics
        print(f"\n📏 SIZE STATISTICS:")
        print(f"  Valid files total size: {stats['total_valid_size']:,} bytes")
        if stats['valid_count'] > 0:
            avg_valid_size = stats['total_valid_size'] / stats['valid_count']
            print(f"  Average valid file size: {avg_valid_size:,.0f} bytes")
        
        if stats['total_invalid_size'] > 0:
            print(f"  Invalid files total size: {stats['total_invalid_size']:,} bytes")
            if stats['invalid_count'] > 0:
                avg_invalid_size = stats['total_invalid_size'] / stats['invalid_count']
                print(f"  Average invalid file size: {avg_invalid_size:,.0f} bytes")


def validate_file_content(file_path: str, min_file_size: int = 100) -> Tuple[bool, str]:
    """
    Standalone function to validate a single file's content.
    
    Args:
        file_path: Path to the file to validate
        min_file_size: Minimum file size in bytes (default: 100)
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    validator = FileContentValidator(min_file_size=min_file_size)
    is_valid, message, _ = validator.validate_file(file_path)
    return is_valid, message


def validate_directory_files(directory_path: str, min_file_size: int = 100, 
                           file_extensions: Optional[list] = None) -> Dict[str, Any]:
    """
    Validate all files in a directory.
    
    Args:
        directory_path: Path to the directory to scan
        min_file_size: Minimum file size in bytes (default: 100)  
        file_extensions: List of file extensions to check (default: ['.csv'])
        
    Returns:
        Dictionary with validation results
    """
    if file_extensions is None:
        file_extensions = ['.csv']
        
    validator = FileContentValidator(min_file_size=min_file_size)
    
    # Find all files with specified extensions
    file_paths = []
    for ext in file_extensions:
        pattern = f"*{ext}"
        file_paths.extend(Path(directory_path).glob(pattern))
    
    # Convert to string paths
    file_paths = [str(path) for path in file_paths]
    
    # Validate files
    results = validator.validate_files_batch(file_paths)
    
    # Print summary
    validator.print_validation_summary(results)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python file_content_validator.py <file_or_directory_path> [min_size_bytes]")
        print("  file_or_directory_path: Path to file or directory to validate")
        print("  min_size_bytes: Minimum file size in bytes (default: 100)")
        sys.exit(1)
    
    path = sys.argv[1]
    min_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    if os.path.isfile(path):
        # Validate single file
        is_valid, message = validate_file_content(path, min_size)
        print(f"File: {os.path.basename(path)}")
        print(f"Result: {'✅ VALID' if is_valid else '❌ INVALID'}")
        print(f"Message: {message}")
    elif os.path.isdir(path):
        # Validate directory
        results = validate_directory_files(path, min_size)
        print(f"\n📁 Directory: {path}")
        print(f"Processed {results['total_files']} files")
    else:
        print(f"❌ Error: Path '{path}' does not exist or is not accessible")
        sys.exit(1)
