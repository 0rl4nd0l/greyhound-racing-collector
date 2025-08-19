// Null-safe sorting and display logic example
// This demonstrates the exact pattern requested in Step 4

// Convert race time to comparable format (replaces convertToComparableTime function)
function convertToComparableTime(raceTime) {
    if (!raceTime) return "TBD";
    try {
        const timeStr = String(raceTime).trim();
        if (timeStr.match(/\d{1,2}:\d{2}\s*[APap][Mm]/)) {
            // Parse 12-hour format (e.g., "6:31 PM")
            const [time, period] = timeStr.split(/\s+/);
            const [hours, minutes] = time.split(':').map(Number);
            let hour24 = hours;
            if (period.toUpperCase() === 'PM' && hours !== 12) {
                hour24 += 12;
            } else if (period.toUpperCase() === 'AM' && hours === 12) {
                hour24 = 0;
            }
            return hour24 * 60 + minutes;
        } else if (timeStr.match(/\d{1,2}:\d{2}/)) {
            // Parse 24-hour format (e.g., "18:31")
            const [hours, minutes] = timeStr.split(':').map(Number);
            return hours * 60 + minutes;
        }
    } catch (e) {
        console.warn('Error parsing race time:', raceTime, e);
    }
    return "TBD";
}

// Null-safe helpers using nullish coalescing
const safeVenue = v => (v ?? '');      // nullish coalesce
const safeGrade = g => (g ?? '');      // nullish coalesce for grade
const safeDistance = d => (d ?? '');   // nullish coalesce for distance

// Example function demonstrating the refactored sort callback as specified in the task
function sortDateRaces(dateRaces) {
    if (!Array.isArray(dateRaces)) {
        console.warn('dateRaces is not an array:', dateRaces);
        return [];
    }

    // Refactored sort callback with null-safe logic
    dateRaces.sort((a, b) => {
        const tA = convertToComparableTime(a.race_time);
        const tB = convertToComparableTime(b.race_time);
        
        // Handle "TBD" values - put them at the end
        if (tA === "TBD" && tB === "TBD") {
            return safeVenue(a.venue).localeCompare(safeVenue(b.venue));
        }
        if (tA === "TBD") return 1; // TBD values go to end
        if (tB === "TBD") return -1; // TBD values go to end
        
        // Both are numeric, sort normally
        if (tA !== tB) return tA - tB;
        return safeVenue(a.venue).localeCompare(safeVenue(b.venue));
    });

    return dateRaces;
}

// Function to display race data with null-safe helpers
function displayRaceData(race) {
    return {
        venue: safeVenue(race.venue),
        grade: safeGrade(race.grade),
        distance: safeDistance(race.distance),
        race_time: race.race_time ?? '',
        race_name: race.race_name ?? '',
        // Add other fields with null safety as needed
    };
}

// Function to render race table with null-safe display
function renderRaceTable(dateRaces) {
    if (!Array.isArray(dateRaces) || dateRaces.length === 0) {
        return '<tr><td colspan="6" class="text-center">No races found.</td></tr>';
    }

    return dateRaces.map(race => {
        const safeRace = displayRaceData(race);
        return `
            <tr>
                <td>${safeRace.race_name}</td>
                <td>${safeRace.venue}</td>
                <td>${safeRace.race_time}</td>
                <td>${safeRace.distance ? safeRace.distance + 'm' : ''}</td>
                <td>${safeRace.grade}</td>
                <td>
                    <button class="btn btn-sm btn-primary" data-race-id="${race.race_id ?? ''}">
                        Predict
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

// Example usage demonstrating the complete null-safe workflow
function processRaceData(rawDateRaces) {
    console.log('Processing race data with null-safe sorting...');
    
    // Sort races with null-safe logic
    const sortedRaces = sortDateRaces([...rawDateRaces]); // Create a copy to avoid mutating original
    
    console.log('Races sorted successfully:', sortedRaces.length);
    
    // Render with null-safe display
    const tableHtml = renderRaceTable(sortedRaces);
    
    // Update DOM if table element exists
    const tableBody = document.getElementById('races-table-body');
    if (tableBody) {
        tableBody.innerHTML = tableHtml;
    }
    
    return sortedRaces;
}

// Export functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        convertToComparableTime,
        safeVenue,
        safeGrade,
        safeDistance,
        sortDateRaces,
        displayRaceData,
        renderRaceTable,
        processRaceData
    };
}

// Make functions available globally for browser usage
if (typeof window !== 'undefined') {
    window.NullSafeSorting = {
        convertToComparableTime,
        safeVenue,
        safeGrade,
        safeDistance,
        sortDateRaces,
        displayRaceData,
        renderRaceTable,
        processRaceData
    };
}
