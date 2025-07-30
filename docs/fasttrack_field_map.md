# FastTrack Site Structure and Field Map

This document outlines the URL structure, API endpoints, and data field mappings for the FastTrack website.

## URL Patterns

*   **Dog Profile URL:** `https://fasttrack.grv.org.au/Dog/Form/{DogId}`
*   **Race Meeting URL:** `https://fasttrack.grv.org.au/RaceField/ViewRaces/{MeetingId}`
*   **Race Result URL:** `https://fasttrack.grv.org.au/RaceField/ViewRaces/{MeetingId}?raceId={RaceId}`

## API Endpoints

*   **Dog Profile API (Form Data):** `https://fasttrack.grv.org.au/Dog/FormContent/{DogId}?key={ApiKey}&sharedKey={SharedKey}`
    *   This endpoint returns JSON containing HTML content of the dog's form guide.
    *   Keys are dynamic and must be extracted from the main dog profile page's JavaScript.
*   **Race Meetings/Results (Static HTML):** Direct HTTP requests to race URLs return full HTML pages.

## Static vs Dynamic Content

*   **Static Content:** Race meeting pages and race result pages return full HTML content via direct HTTP requests.
*   **Dynamic Content:** Dog form data requires dynamic key/sharedKey parameters extracted from JavaScript.

## Field Mappings

### Race Result Data (from HTML pages)

| Field Name | HTML Selector/Location | Description |
|---|---|---|
| Dog Name | `.ReportRace a[rel="dog-summary-link"]` | Dog name with link to profile |
| Box Number | `.ReportRaceDogRugNumber` | Starting box number (1-8) |
| Trainer | `.ReportRace` (trainer cell) | Trainer name and location |
| Odds | `.ReportRace` (odds cell) | Fixed odds for the dog |
| Prizemoney | `.ReportRace` (prize cell) | Prize money earned |
| Weight | Dog form details | Race weight |
| Placing | Dog form details | Final placing in race |
| Time | Dog form details | Race time |
| Margin | Dog form details | Margin from winner |
| Split Time | Dog form details | First split time |
| PIR | Dog form details | Performance Index Rating |
| Video Link | `a[href*="RaceVideo"]` | Link to race video |
| Sire/Dam | `.ReportRaceDogFormMainDetails` | Breeding information |
| Winning Boxes | `.WinBoxTable` | Historical winning box statistics |

### Dog Data (from FormContent HTML)

| Field Name | HTML Selector | Description |
|---|---|---|
| `Pl` | `td` | Placing in the race. |
| `Bx` | `td` | Box number. |
| `Wght` | `td` | Weight of the dog. |
| `Dist` | `td` | Race distance. |
| `Trk` | `td` | Track code. |
| `Race` | `td > a` | Race number and link to race details. |
| `Date` | `td` | Date of the race. |
| `Time` | `td` | The dog's race time. |
| `BON` | `td` | Best of night time. |
| `M'gin` | `td` | Margin from the winner. |
| `1st/2nd Time` | `td` | Time of the 1st or 2nd placed dog. |
| `Split1` | `td` | First split time. |
| `PIR` | `td` | Points in running. |
| `Comment` | `td` | Race comment. |
| `S/P` | `td` | Starting price. |
| `Grade` | `td` | Race grade. |
| `Hcap` | `td` | Handicap. |
| `Video` | `td > a` | Link to the race video. |

