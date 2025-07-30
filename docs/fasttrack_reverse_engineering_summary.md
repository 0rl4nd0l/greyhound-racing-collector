# FastTrack Reverse-Engineering Summary

## Overview
Successfully completed reverse-engineering of the FastTrack greyhound racing website structure, identifying URL patterns, API endpoints, and data field mappings.

## Key Findings

### URL Structure
- **Dog Profile**: `https://fasttrack.grv.org.au/Dog/Form/{DogId}` (e.g., 890320106)
- **Race Meeting**: `https://fasttrack.grv.org.au/RaceField/ViewRaces/{MeetingId}` (e.g., 1163670701)
- **Race Result**: `https://fasttrack.grv.org.au/RaceField/ViewRaces/{MeetingId}?raceId={RaceId}` (e.g., 1148743411?raceId=1186391057)

### API Endpoints
1. **Dog Form Data**: `https://fasttrack.grv.org.au/Dog/FormContent/{DogId}?key={ApiKey}&sharedKey={SharedKey}`
   - Returns JSON with HTML content
   - Keys are dynamically generated and must be intercepted from network requests
   - Contains historical race data for the dog

### Content Types
- **Static HTML**: Race meeting and result pages can be scraped with direct HTTP requests
- **Dynamic AJAX**: Dog form data requires dynamic authentication keys

### Data Richness
The FastTrack site provides extremely rich data including:
- Detailed race results with times, margins, and placings
- Dog breeding information (sire/dam)
- Trainer details and locations
- Fixed odds and prizemoney
- Video links for race replays
- Performance Index Ratings (PIR)
- Split times and sectional data
- Historical winning box statistics

## Artifacts Created
1. `docs/fasttrack_field_map.md` - Complete field mapping documentation
2. `samples/fasttrack_raw/` - Sample data files:
   - `dog_890320106_form_content.html` - Raw dog form data
   - `dog_890320106_races.csv` - Parsed race history
   - `race_result_1186391057.html` - Race result page
   - `race_meeting_1163670701.html` - Race meeting page
   - `fasttrack.grv.org.au.har` - Network traffic analysis
   - `form_content_url.txt` - Intercepted API URL

## Technical Implementation
- Used Playwright for headless browser automation
- Implemented network request interception to capture dynamic API URLs
- Created parsing scripts to extract structured data from HTML/JSON responses
- Successfully extracted 18+ data fields per dog race record

## Next Steps
This reverse-engineering provides the foundation for:
1. Building automated scrapers for FastTrack data
2. Integrating FastTrack data into existing prediction systems
3. Comparing data quality with existing sources (thedogs.com.au)
4. Developing real-time data collection pipelines
