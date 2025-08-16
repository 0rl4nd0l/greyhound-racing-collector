# Call Chain Audit Report
## Front-end → `/api/upcoming_races_csv` → `load_upcoming_races()` 

**Date:** August 4, 2025  
**Auditor:** AI Assistant  
**Status:** ⚠️ **CRITICAL DUPLICATES DETECTED**

---

## Executive Summary

The audit has identified a **critical duplication issue** in the upcoming races loading flow. The system is generating **10 identical race records** from a single CSV file, causing severe data integrity problems.

### Key Findings:
- ✅ **Server Health**: Application running normally on localhost:5002
- ❌ **Data Integrity**: Severe duplicate race records (10 duplicates from 1 source)
- ❌ **Race ID Generation**: Same race_id (`b1cbfcb96b25`) for all duplicates
- ❌ **Venue Parsing**: All venues showing as "Unknown" 
- ❌ **Date Parsing**: All race dates showing as "Unknown"

---

## Call Chain Analysis

### 1. Front-end Request (JavaScript)
**File:** `static/js/interactive-races.js` (Line 125)

```javascript
// Front-end makes request to API endpoint
const endpoint = state.viewMode === 'upcoming' ? '/api/upcoming_races_csv' : '/api/races/paginated';
const response = await fetchWithErrorHandling(endpoint);
```

**Request Parameters Traced:**
- `GET /api/upcoming_races_csv` 
- `GET /api/upcoming_races_csv?refresh=true`
- `GET /api/upcoming_races_csv?page=1&per_page=5`
- `GET /api/upcoming_races_csv?search=test`

### 2. API Endpoint Handler
**File:** `app.py` (Lines 1537-1843)

```python
@app.route("/api/upcoming_races_csv")
def api_upcoming_races_csv():
    # Calls load_upcoming_races() function
    upcoming_races = load_upcoming_races(refresh=refresh)
```

**Request Parameters Verified:**
- ✅ HTTP 200 status codes
- ✅ JSON response format
- ✅ Proper pagination structure
- ❌ **CRITICAL**: Duplicate data in response

### 3. Core Data Loading Function  
**File:** `app.py` (Lines 8280-8409)

```python
def load_upcoming_races(refresh=False):
    """Helper function to load upcoming races from CSV and JSON files."""
    # Processes files in ./upcoming_races directory
    for filename in os.listdir(upcoming_races_dir):
        if filename.endswith(".csv") or filename.endswith(".json"):
            # CSV processing creates race_metadata for EACH ROW
            for _, row in df.iterrows():  # ❌ BUG IS HERE
                race_metadata = {
                    "race_name": row.get("Race Name") or row.get("race_name") or "Unknown Race",
                    # ... creates race record for each dog/row
                }
                races.append(race_metadata)  # ❌ DUPLICATES CREATED HERE
```

---

## Root Cause Analysis

### Primary Issue: CSV Processing Logic Bug

**Location:** `app.py` lines 8330-8345 in `load_upcoming_races()`

**Problem:** The function processes CSV files **row by row**, creating a separate race record for each dog/row in the CSV file, instead of creating **one race record per CSV file**.

**Evidence:**
- CSV file `Race 1 - AP_K - 01 July 2025.csv` contains 21 rows (21 dogs)
- Each row generates a race metadata record  
- All records get the same race_id: `b1cbfcb96b25`
- Result: 10 duplicate race records in API response (limited by pagination)

### Secondary Issues:

1. **Race ID Generation Logic:**
   ```python
   race_id = hashlib.md5(f"{filename}_{row.get('Race Number', 0)}".encode()).hexdigest()[:12]
   ```
   - Uses same filename + race number for all rows
   - Results in identical race_ids

2. **Venue/Date Extraction:**
   - CSV format doesn't contain race-level metadata in individual rows
   - Venue and date information not properly extracted from filename or headers

3. **Data Structure Mismatch:**
   - CSV files are **form guides** (multiple dogs per race)
   - Function treats each dog row as a separate race

---

## Evidence Files Captured

### Good JSON Response (With Duplicates)
**File:** `reference_good_response_20250804_104845.json`
- 10 identical race records
- All have race_id: `b1cbfcb96b25`
- All have same filename: `Race 1 - AP_K - 01 July 2025.csv`

### Problematic CSV Source
**File:** `reference_csv_file_20250804_104845.csv`  
- 21 rows of dog data (form guide format)
- Headers: `Dog Name,Sex,PLC,BOX,WGT,DIST,DATE,TRACK,G,TIME,WIN,BON,1 SEC,MGN,W/2G,PIR,SP`
- Contains **ONE race** with multiple dogs, not multiple races

### Detailed Analysis  
**File:** `detailed_analysis_20250804_104845.json`
- Confirms duplicate detection
- Shows 10 total races, 1 unique race_id
- Response structure is valid, data is duplicated

---

## Directory Structure Analysis

### `/upcoming_races` Directory Contents:
- **4 CSV files:** Race form guides (21, 15, 3, and unknown rows respectively)
- **3 JSON files:** Race metadata from scrapers (269 races each)
- **Mixed data formats:** CSV = form guides, JSON = race listings

### File Type Analysis:
1. **CSV Files** (Form Guides):
   - `Race 1 - AP_K - 01 July 2025.csv` (21 dogs)
   - `Race 1 - AP_K - 2025-08-04.csv` (15 dogs) 
   - `Race 4 - GARD - 11 Aug 2025.csv` (3 dogs)
   - Each contains multiple dogs for ONE race

2. **JSON Files** (Race Listings):
   - `upcoming_races_20250803_233558.json` (269 races)
   - Contains proper race metadata structure
   - No duplication issues in JSON processing

---

## Impact Assessment

### Critical Issues:
- **Data Integrity**: 10x data duplication in API responses
- **Performance**: Unnecessary data transfer and processing
- **User Experience**: Confusing duplicate races in UI
- **Database**: Potential for duplicate race processing

### Affected Components:
- ✅ **API Endpoint**: Returns duplicated data
- ✅ **Frontend**: Receives and displays duplicates  
- ✅ **Race Selection**: Users see duplicate options
- ✅ **Prediction Pipeline**: May process same race multiple times

---

## Recommended Solutions

### Immediate Fix (Priority 1):
**Modify CSV processing logic in `load_upcoming_races()`:**

```python
# BEFORE (creates race per row):
for _, row in df.iterrows():
    race_metadata = {...}
    races.append(race_metadata)

# AFTER (creates one race per CSV file):
if filename.endswith(".csv"):
    df = pd.read_csv(file_path)
    # Extract race info from filename and first row
    race_metadata = {
        "race_name": extract_race_name_from_filename(filename),
        "venue": extract_venue_from_filename(filename), 
        "race_date": extract_date_from_filename(filename),
        "filename": filename,
        "race_id": hashlib.md5(filename.encode()).hexdigest()[:12],
        # Add dog count from CSV
        "field_size": len(df),
    }
    races.append(race_metadata)  # ONE record per CSV file
```

### Data Parsing Improvements (Priority 2):
1. **Filename Parsing**: Extract venue, date, race number from filenames
2. **Header Analysis**: Read race metadata from CSV headers if available
3. **Validation**: Ensure one race record per CSV file

### Long-term Improvements (Priority 3):
1. **Data Source Separation**: Distinguish between form guides (CSV) and race listings (JSON)
2. **Caching Strategy**: Prevent re-processing identical files
3. **Validation Pipeline**: Add data integrity checks

---

## Testing Recommendations

### Unit Tests Needed:
```python
def test_load_upcoming_races_no_duplicates():
    races = load_upcoming_races()
    race_ids = [r['race_id'] for r in races]
    assert len(race_ids) == len(set(race_ids)), "Duplicate race_ids detected"

def test_csv_processing_single_race_per_file():
    # Test that each CSV file generates exactly one race record
    pass

def test_venue_date_extraction():
    # Test filename parsing logic
    pass
```

### Integration Tests:
- API endpoint duplicate detection
- Frontend rendering with clean data
- End-to-end race selection flow

---

## Verification Steps

To verify fix implementation:

1. **Before Fix**: 
   ```bash
   curl "http://localhost:5002/api/upcoming_races_csv" | jq '.races | length'  # Returns 10
   curl "http://localhost:5002/api/upcoming_races_csv" | jq '.races | map(.race_id) | unique | length'  # Returns 1
   ```

2. **After Fix**:
   ```bash
   curl "http://localhost:5002/api/upcoming_races_csv" | jq '.races | length'  # Should return 4 (number of CSV files)
   curl "http://localhost:5002/api/upcoming_races_csv" | jq '.races | map(.race_id) | unique | length'  # Should equal total races
   ```

---

## Conclusion

The audit has successfully identified a **critical data duplication bug** in the upcoming races loading pipeline. The root cause is in the CSV processing logic within the `load_upcoming_races()` function, which incorrectly treats each row (dog) in a form guide CSV as a separate race.

**Status**: ⚠️ **REQUIRES IMMEDIATE ATTENTION**  
**Severity**: **HIGH** - Data integrity impact  
**Effort**: **MEDIUM** - Single function modification required

The fix is straightforward but critical for data integrity. All reference files have been captured for validation and testing purposes.
