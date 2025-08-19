#!/bin/bash

echo "========================================================================="
echo "GUNICORN MONITORING - SAFER RUNTIME LIMITS"
echo "Started at: $(date)"
echo "Monitoring configuration:"
echo "  - Workers: 4"
echo "  - Timeout: 90s"
echo "  - Max requests per worker: 500"  
echo "  - Worker class: sync"
echo "  - Environment:"
echo "    - FLASK_PROFILING=off"
echo "    - OPTIMIZATION_ENABLED=true" 
echo "========================================================================="

for i in {1..10}; do
    echo ""
    echo "--- MINUTE $i of 10 ---"
    echo "Time: $(date)"
    
    # Check if gunicorn processes are running
    GUNICORN_PROCESSES=$(ps aux | grep gunicorn | grep -v grep | wc -l)
    echo "Gunicorn processes running: $GUNICORN_PROCESSES"
    
    if [ $GUNICORN_PROCESSES -eq 0 ]; then
        echo "⚠️  WARNING: No gunicorn processes found!"
        break
    fi
    
    # Show detailed process info
    echo "Process details:"
    ps aux | grep gunicorn | grep -v grep | awk '{printf "  PID: %s, CPU: %s%%, MEM: %s%%, COMMAND: %s\n", $2, $3, $4, substr($0, index($0,$11))}'
    
    # Test endpoint responsiveness
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5002/ 2>/dev/null)
    if [ "$HTTP_STATUS" = "200" ]; then
        echo "✅ Application responding: HTTP $HTTP_STATUS"
    else
        echo "❌ Application not responding: HTTP $HTTP_STATUS"
    fi
    
    # Show memory usage summary
    echo "Memory usage summary:"
    ps aux | grep gunicorn | grep -v grep | awk '{sum_mem += $6} END {printf "  Total RSS: %.2f MB\n", sum_mem/1024}'
    
    # Show system load
    echo "System load: $(uptime | awk -F'load averages:' '{print $2}')"
    
    if [ $i -lt 10 ]; then
        sleep 60
    fi
done

echo ""
echo "========================================================================="
echo "MONITORING COMPLETED at $(date)"
echo "Final status:"
FINAL_PROCESSES=$(ps aux | grep gunicorn | grep -v grep | wc -l)
echo "  Gunicorn processes: $FINAL_PROCESSES"
if [ $FINAL_PROCESSES -gt 0 ]; then
    echo "✅ Gunicorn is running successfully with safer runtime limits!"
else
    echo "❌ Gunicorn processes not found"
fi
echo "========================================================================="
