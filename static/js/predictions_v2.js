
document.addEventListener('DOMContentLoaded', () => {
    const raceCheckboxes = document.querySelectorAll('.race-checkbox');
    const bulkActionsSelect = document.getElementById('bulk-actions');
    const runBulkActionsBtn = document.getElementById('run-bulk-actions');
    const exportCsvBtn = document.getElementById('export-csv');
    const exportPdfBtn = document.getElementById('export-pdf');
    const autoRefreshToggle = document.getElementById('auto-refresh-toggle');
    const autoRefreshInterval = document.getElementById('auto-refresh-interval');
    const runAllPredictionsBtn = document.getElementById('run-all-predictions');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');

    let autoRefreshTimer;

    function getSelectedRaceIds() {
        const selectedIds = [];
        raceCheckboxes.forEach(checkbox => {
            if (checkbox.checked) {
                selectedIds.push(checkbox.dataset.raceId);
            }
        });
        return selectedIds;
    }

    function runBulkPredictions(raceIds) {
        let completed = 0;
        const total = raceIds.length;
        
        raceIds.forEach(raceId => {
            // Simulate prediction run
            setTimeout(() => {
                completed++;
                const progress = (completed / total) * 100;
                progressBar.style.width = `${progress}%`;
                progressText.innerText = `Processed ${completed} of ${total}`;
                if (completed === total) {
                    progressText.innerText = 'Completed all predictions.';
                }
            }, Math.random() * 2000);
        });
    }

    function exportToCsv() {
        const data = [['Race ID', 'Dog', 'Win Probability']];
        // Populate data from the table
        const tableRows = document.querySelectorAll('#predictions-table tbody tr');
        tableRows.forEach(row => {
            const raceId = row.cells[0].innerText;
            const dog = row.cells[1].innerText;
            const winProb = row.cells[2].innerText;
            data.push([raceId, dog, winProb]);
        });

        const csv = Papa.unparse(data);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.setAttribute('download', 'predictions.csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    function exportToPdf() {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        
        doc.autoTable({ html: '#predictions-table' });
        doc.save('predictions.pdf');
    }

    function handleAutoRefresh() {
        if (autoRefreshToggle.checked) {
            const interval = parseInt(autoRefreshInterval.value, 10) * 1000;
            localStorage.setItem('autoRefreshInterval', interval);
            autoRefreshTimer = setInterval(() => {
                // Add logic to refresh predictions data
                console.log('Auto-refreshing predictions...');
            }, interval);
        } else {
            clearInterval(autoRefreshTimer);
        }
        localStorage.setItem('autoRefreshEnabled', autoRefreshToggle.checked);
    }

    runBulkActionsBtn.addEventListener('click', () => {
        const action = bulkActionsSelect.value;
        const selectedIds = getSelectedRaceIds();

        if (action === 'run-predictions' && selectedIds.length > 0) {
            runBulkPredictions(selectedIds);
        }
    });

    if(exportCsvBtn) {
        exportCsvBtn.addEventListener('click', exportToCsv);
    }

    if(exportPdfBtn) {
        exportPdfBtn.addEventListener('click', exportToPdf);
    }
    
    autoRefreshToggle.addEventListener('change', handleAutoRefresh);
    autoRefreshInterval.addEventListener('change', handleAutoRefresh);
    
    if(runAllPredictionsBtn) {
        runAllPredictionsBtn.addEventListener('click', () => {
            const allRaceIds = Array.from(raceCheckboxes).map(cb => cb.dataset.raceId);
            runBulkPredictions(allRaceIds);
        });
    }

    // Load saved auto-refresh settings
    const savedInterval = localStorage.getItem('autoRefreshInterval');
    const savedEnabled = localStorage.getItem('autoRefreshEnabled');

    if (savedInterval) {
        autoRefreshInterval.value = savedInterval / 1000;
    }
    if (savedEnabled === 'true') {
        autoRefreshToggle.checked = true;
        handleAutoRefresh();
    }
});
