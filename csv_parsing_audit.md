# CSV Parsing Audit

## Overview
This document provides a comprehensive audit of the CSV parsing logic against the FORM_GUIDE_SPEC.md specification.

## Current Implementation

1. **Header Detection**
   - Implemented with drift detection based on a 30% threshold from `expected_headers`.

2. **Continuation Rows**
   - Handled using forward-fill logic for dog names and box numbers.

3. **Delimiters & BOM**
   - Mixed delimiter detection and BOM removal are implemented.

4. **Invisible Unicode**
   - Detection and cleaning are performed.

5. **Error Handling & Quarantine**
   - Comprehensive error detection with a quarantine mechanism for problematic files.

6. **Validation & Mapping**
   - Column validation and mapping are handled with detail, especially for "Dog Name."

## Gaps & Deviations

- **Forward-Fill Logic:** 
  - Spec-compliant, but robustness in handling edge cases needs verification through extensive tests.

- **Invisible Unicode Characters:**
  - Consistent logging required for cleaned characters to aid debugging.

- **File Integrity & Recovery:**
  - Limited tests on real-world corrupted CSVs.

- **Flexible Parsing:**
  - More real-world flexibility needed for validation interfaces.

## Action Items

1. **Tests for Mixed Delimiters:** 
   - Develop more test cases simulating complex delimiter situations.
  
2. **Improve Continuation Row Handling:**
   - Test extreme cases with extensive continuation rows.
  
3. **Gap Analysis for Non-CSV Formats:**
   - Implement checks for common non-CSV formats encountered in datasets.
  
4. **Enhance Debugging:**
   - Additional logging features to track pre-cleanup states for invisible characters.
  
5. **Validation Enhancements:**
   - Strengthen the validation mechanism by considering fuzzy logic for "Dog Name" header detection.

