# Form Guide Parsing Specification

## Overview
This document defines the standard format and parsing rules for greyhound racing form guides, including common pitfalls and detection strategies.

## Dog Block Structure

### Standard Dog Block Format
Each dog entry consists of:
```
Dog Name, Box Number, Trainer, Weight, Form, Recent Times, etc.
```

### Example Dog Block
```csv
Lightning Bolt,1,J. Smith,32.5,12341,29.45-29.67-30.12,Good
Speedy Susan,2,M. Jones,30.8,23154,29.78-30.01-29.89,Fast
Thunder Strike,3,K. Brown,33.2,31452,30.15-29.95-30.33,Slow
```

### Blank Continuation Rows
Some form guides use blank continuation rows for additional dog information:
```csv
Lightning Bolt,1,J. Smith,32.5,12341,29.45-29.67-30.12,Good
,,,,,Additional comments or sectional times,
Speedy Susan,2,M. Jones,30.8,23154,29.78-30.01-29.89,Fast
```

### Forward-Fill Rule
When encountering blank cells in continuation rows:
1. **Dog Name**: Forward-fill from previous non-blank dog name
2. **Box Number**: Forward-fill from previous non-blank box number
3. **Other Fields**: Use blank/null values unless explicitly continued

## Common Pitfalls and Detection

### 1. Mixed Delimiters
**Problem**: Inconsistent use of commas, semicolons, or tabs
```csv
Lightning Bolt,1,J. Smith;32.5,12341	29.45-29.67-30.12,Good
```

**Detection Regex**:
```python
import re
mixed_delimiter_pattern = r'[,;\t]{2,}|,[;\t]|[;\t],'
if re.search(mixed_delimiter_pattern, line):
    warnings.append("Mixed delimiters detected")
```

### 2. Invisible Unicode Characters
**Problem**: Hidden unicode characters (BOM, zero-width spaces)
```python
# Detection
import unicodedata
def has_invisible_chars(text):
    for char in text:
        if unicodedata.category(char) in ['Cf', 'Cc']:  # Format/Control chars
            return True
    return False
```

### 3. Header Drift
**Problem**: Headers shifting position due to added/removed columns

**Detection Strategy**:
```python
def detect_header_drift(headers, expected_headers):
    """Check if headers match expected positions"""
    drift_score = 0
    for i, (actual, expected) in enumerate(zip(headers, expected_headers)):
        if actual.strip().lower() != expected.strip().lower():
            drift_score += 1
    
    return drift_score > len(expected_headers) * 0.3  # 30% drift threshold
```

### 4. Embedded Newlines in Fields
**Problem**: Multi-line data within CSV fields
```csv
"Lightning Bolt
(Also known as LB)",1,J. Smith,32.5
```

**Detection**:
```python
def has_embedded_newlines(field):
    return '\n' in field or '\r' in field
```

## Validation Rules

### Required Fields
- Dog Name (non-empty string)
- Box Number (integer 1-8)
- Weight (float > 20.0, < 40.0)

### Optional Fields
- Trainer Name
- Form String (typically 5 digits)
- Recent Times (dash-separated floats)
- Track Condition

### Data Type Validation
```python
def validate_dog_block(dog_data):
    errors = []
    
    # Dog name validation
    if not dog_data.get('name') or not isinstance(dog_data['name'], str):
        errors.append("Invalid dog name")
    
    # Box number validation
    box_num = dog_data.get('box_number')
    if not isinstance(box_num, int) or box_num < 1 or box_num > 8:
        errors.append("Invalid box number")
    
    # Weight validation
    weight = dog_data.get('weight')
    if weight and (not isinstance(weight, (int, float)) or weight < 20.0 or weight > 40.0):
        errors.append("Invalid weight")
    
    return errors
```

## Parsing Algorithm

### Step 1: Pre-processing
```python
def preprocess_csv(content):
    # Remove BOM
    content = content.replace('\ufeff', '')
    
    # Normalize line endings
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    
    # Check for mixed delimiters
    check_mixed_delimiters(content)
    
    return content
```

### Step 2: Header Detection
```python
def detect_headers(first_line):
    potential_headers = first_line.split(',')
    
    # Known header patterns
    header_patterns = {
        'dog_name': r'(dog|name|greyhound)',
        'box_number': r'(box|trap|number)',
        'trainer': r'(trainer|handler)',
        'weight': r'(weight|kg)',
        'form': r'(form|recent)'
    }
    
    return match_headers(potential_headers, header_patterns)
```

### Step 3: Dog Block Extraction
```python
def extract_dog_blocks(lines, headers):
    dogs = []
    current_dog = None
    
    for line in lines[1:]:  # Skip header
        fields = parse_csv_line(line)
        
        if is_new_dog_block(fields):
            if current_dog:
                dogs.append(current_dog)
            current_dog = create_dog_from_fields(fields, headers)
        else:
            # Continuation row
            if current_dog:
                merge_continuation_data(current_dog, fields, headers)
    
    if current_dog:
        dogs.append(current_dog)
    
    return dogs
```

## Error Handling

### Quarantine Conditions
Move files to quarantine if:
1. More than 50% of rows fail validation
2. No valid dog blocks detected
3. Critical fields missing (name, box number)
4. Encoding errors persist after cleanup attempts

### Recovery Strategies
1. **Encoding Issues**: Try UTF-8, Latin-1, CP1252 in sequence
2. **Delimiter Issues**: Auto-detect most common delimiter
3. **Header Issues**: Use positional fallback if header matching fails
4. **Missing Data**: Use form guide defaults where appropriate

## Testing Examples

### Valid Form Guide
```csv
Dog Name,Box,Trainer,Weight,Form,Times,Condition
Lightning Bolt,1,J. Smith,32.5,12341,29.45-29.67,Good
Speedy Susan,2,M. Jones,30.8,23154,29.78-30.01,Fast
```

### Invalid Form Guide (Mixed Delimiters)
```csv
Dog Name,Box;Trainer,Weight	Form,Times,Condition
Lightning Bolt,1;J. Smith,32.5	12341,29.45-29.67,Good
```

### Edge Case (Continuation Rows)
```csv
Dog Name,Box,Trainer,Weight,Form,Times,Condition
Lightning Bolt,1,J. Smith,32.5,12341,29.45-29.67,Good
,,,,,Additional sectional: 5.23,
Speedy Susan,2,M. Jones,30.8,23154,29.78-30.01,Fast
```

## Implementation Notes

- Always validate data types after parsing
- Log all parsing warnings for debugging
- Maintain parsing statistics (success rate, common errors)
- Use conservative defaults when data is ambiguous
- Preserve original file for manual review if parsing fails
