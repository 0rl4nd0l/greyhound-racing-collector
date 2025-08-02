#!/bin/bash
"""
Setup Daily Schema Drift Monitoring Cron Job

This script sets up a daily cron job to monitor production database schema drift.
Run this on your production server to enable automatic schema monitoring.

Usage:
    bash scripts/setup_schema_monitoring_cron.sh --prod-db-url="postgresql://..." [options]
"""

set -e

# Default values
PROD_DB_URL=""
ALERT_WEBHOOK=""
EMAIL_SMTP_SERVER=""
EMAIL_USERNAME=""
EMAIL_PASSWORD=""
EMAIL_TO=""
CRON_TIME="0 6 * * *"  # Daily at 6 AM
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_FILE="${PROJECT_ROOT}/logs/schema_monitoring_setup.log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prod-db-url)
            PROD_DB_URL="$2"
            shift 2
            ;;
        --alert-webhook)
            ALERT_WEBHOOK="$2"
            shift 2
            ;;
        --email-smtp-server)
            EMAIL_SMTP_SERVER="$2"
            shift 2
            ;;
        --email-username)
            EMAIL_USERNAME="$2"
            shift 2
            ;;
        --email-password)
            EMAIL_PASSWORD="$2"
            shift 2
            ;;
        --email-to)
            EMAIL_TO="$2"
            shift 2
            ;;
        --cron-time)
            CRON_TIME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --prod-db-url=URL [options]"
            echo ""
            echo "Options:"
            echo "  --prod-db-url        Production database URL (required)"
            echo "  --alert-webhook      Webhook URL for alerts (optional)"
            echo "  --email-smtp-server  SMTP server for email alerts (optional)"
            echo "  --email-username     Email username (optional)"
            echo "  --email-password     Email password (optional)"
            echo "  --email-to           Email recipient (optional)"
            echo "  --cron-time          Cron schedule (default: '0 6 * * *' - daily at 6 AM)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$PROD_DB_URL" ]]; then
    echo "Error: --prod-db-url is required"
    exit 1
fi

# Create logs directory
mkdir -p "${PROJECT_ROOT}/logs"

# Setup logging
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

echo "=================================================="
echo "Schema Drift Monitoring Setup"
echo "Started: $(date)"
echo "Project Root: $PROJECT_ROOT"
echo "=================================================="

# Check if Python environment is set up
echo "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if required Python packages are installed
echo "Checking required Python packages..."
python3 -c "import sqlalchemy, pytest, requests" 2>/dev/null || {
    echo "Error: Required Python packages not found. Please install requirements:"
    echo "pip install -r requirements.txt -r requirements-test.txt"
    exit 1
}

# Test database connection
echo "Testing database connection..."
python3 -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from tests.test_database_schema_consistency import DatabaseSchemaConsistencyTester
try:
    tester = DatabaseSchemaConsistencyTester('${PROD_DB_URL}')
    schema_hash = tester.generate_schema_hash()
    print(f'✓ Database connection successful. Schema hash: {schema_hash[:16]}...')
except Exception as e:
    print(f'✗ Database connection failed: {e}')
    sys.exit(1)
" || exit 1

# Create baseline schema snapshot
echo "Creating baseline schema snapshot..."
python3 "${PROJECT_ROOT}/scripts/schema_drift_monitor.py" \
    --prod-db-url="$PROD_DB_URL" \
    --create-baseline || {
    echo "Error: Failed to create baseline schema snapshot"
    exit 1
}

# Build cron command
CRON_COMMAND="cd ${PROJECT_ROOT} && python3 scripts/schema_drift_monitor.py --prod-db-url=\"${PROD_DB_URL}\""

if [[ -n "$ALERT_WEBHOOK" ]]; then
    CRON_COMMAND="${CRON_COMMAND} --alert-webhook=\"${ALERT_WEBHOOK}\""
fi

if [[ -n "$EMAIL_SMTP_SERVER" && -n "$EMAIL_USERNAME" && -n "$EMAIL_PASSWORD" ]]; then
    CRON_COMMAND="${CRON_COMMAND} --email-smtp-server=\"${EMAIL_SMTP_SERVER}\""
    CRON_COMMAND="${CRON_COMMAND} --email-username=\"${EMAIL_USERNAME}\""
    CRON_COMMAND="${CRON_COMMAND} --email-password=\"${EMAIL_PASSWORD}\""
    
    if [[ -n "$EMAIL_TO" ]]; then
        CRON_COMMAND="${CRON_COMMAND} --email-to=\"${EMAIL_TO}\""
    fi
fi

CRON_COMMAND="${CRON_COMMAND} --output-file=\"${PROJECT_ROOT}/logs/schema_drift_\$(date +\\%Y\\%m\\%d).json\" >> ${PROJECT_ROOT}/logs/schema_drift_cron.log 2>&1"

# Full cron entry
CRON_ENTRY="${CRON_TIME} ${CRON_COMMAND}"

echo "Cron command prepared:"
echo "$CRON_ENTRY"
echo ""

# Add to crontab
echo "Adding cron job..."
(crontab -l 2>/dev/null | grep -v "schema_drift_monitor.py"; echo "$CRON_ENTRY") | crontab -

if [[ $? -eq 0 ]]; then
    echo "✓ Cron job added successfully!"
else
    echo "✗ Failed to add cron job"
    exit 1
fi

# Verify cron job
echo "Current crontab entries for schema monitoring:"
crontab -l | grep "schema_drift_monitor.py" || echo "No schema monitoring cron jobs found"

# Create logrotate configuration
echo "Setting up log rotation..."
LOGROTATE_CONFIG="/etc/logrotate.d/schema_drift_monitoring"

if [[ -w "/etc/logrotate.d" ]]; then
    sudo tee "$LOGROTATE_CONFIG" > /dev/null << EOF
${PROJECT_ROOT}/logs/schema_drift_*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 $(whoami) $(whoami)
}

${PROJECT_ROOT}/logs/schema_drift_cron.log {
    weekly
    rotate 12
    compress
    delaycompress
    missingok
    notifempty
    create 644 $(whoami) $(whoami)
}
EOF
    echo "✓ Logrotate configuration created at $LOGROTATE_CONFIG"
else
    echo "⚠ Could not create logrotate configuration (no write access to /etc/logrotate.d)"
    echo "   You may want to manually set up log rotation for:"
    echo "   - ${PROJECT_ROOT}/logs/schema_drift_*.log"
    echo "   - ${PROJECT_ROOT}/logs/schema_drift_cron.log"
fi

# Test the monitoring script
echo "Testing schema monitoring script..."
python3 "${PROJECT_ROOT}/scripts/schema_drift_monitor.py" \
    --prod-db-url="$PROD_DB_URL" \
    --output-file="${PROJECT_ROOT}/logs/schema_drift_test.json" || {
    echo "⚠ Warning: Test run of monitoring script failed. Check configuration."
}

echo "=================================================="
echo "Schema Drift Monitoring Setup Complete!"
echo "Completed: $(date)"
echo ""
echo "Summary:"
echo "- Cron job scheduled: $CRON_TIME"
echo "- Database URL: ${PROD_DB_URL}"
echo "- Baseline snapshot: schema_baseline.json"
echo "- Logs directory: ${PROJECT_ROOT}/logs/"
echo "- Next run: $(date -d 'tomorrow 06:00')"
echo ""
echo "To verify the setup:"
echo "  crontab -l | grep schema_drift"
echo ""
echo "To manually run the monitoring:"
echo "  python3 scripts/schema_drift_monitor.py --prod-db-url=\"$PROD_DB_URL\""
echo "=================================================="
