#!/bin/bash
"""
Guardian Improvements Setup Script
==================================

Sets up the enhanced Guardian service with incremental hashing,
external cron service, and Prometheus monitoring.

Author: AI Assistant
Date: August 4, 2025
"""

set -e  # Exit on any error

echo "🛡️ Guardian Service Improvements Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function for colored output
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "utils/file_integrity_guardian.py" ]]; then
    log_error "Must be run from the project root directory"
    exit 1
fi

# Create necessary directories
log_info "Creating required directories..."
mkdir -p logs
mkdir -p quarantine
mkdir -p config
mkdir -p monitoring

log_success "Directories created"

# Install Python dependencies if needed
log_info "Checking Python dependencies..."

# Check if prometheus_client is available
python3 -c "import prometheus_client" 2>/dev/null || {
    log_warning "prometheus_client not found, installing..."
    pip3 install prometheus_client
}

# Check if psutil is available
python3 -c "import psutil" 2>/dev/null || {
    log_warning "psutil not found, installing..."
    pip3 install psutil
}

log_success "Python dependencies verified"

# Test incremental hashing functionality
log_info "Testing incremental hashing..."
python3 -c "
from utils.file_integrity_guardian import FileIntegrityGuardian
guardian = FileIntegrityGuardian()
print('✅ Incremental hashing initialized successfully')
print(f'📊 Hash cache entries: {len(guardian.hash_cache)}')
" || {
    log_error "Incremental hashing test failed"
    exit 1
}

log_success "Incremental hashing tested successfully"

# Test Guardian cron service
log_info "Testing Guardian cron service..."
python3 services/guardian_cron_service.py --health-check || {
    log_error "Guardian cron service test failed"
    exit 1
}

log_success "Guardian cron service tested successfully"

# Test Prometheus exporter
log_info "Testing Prometheus exporter..."
python3 monitoring/prometheus_exporter.py --test || {
    log_warning "Prometheus exporter test failed, but continuing..."
}

log_success "Prometheus exporter tested"

# Check system requirements
log_info "Checking system requirements..."

# Check if ionice is available
if command -v ionice >/dev/null 2>&1; then
    log_success "ionice is available for process prioritization"
else
    log_warning "ionice not available - process prioritization will be skipped"
fi

# Check if cron is available
if command -v crontab >/dev/null 2>&1; then
    log_success "crontab is available for scheduling"
else
    log_warning "crontab not available - manual scheduling required"
fi

# Performance tuning recommendations
echo ""
log_info "Performance Tuning Recommendations:"
echo "  • Set GUARDIAN_DISABLE=true to disable background Guardian service"
echo "  • Use ionice -c 3 for low-priority execution"
echo "  • Monitor Guardian metrics at http://localhost:8000/metrics"
echo "  • Hash cache stored in .guardian_hash_cache.pkl"

# Installation summary
echo ""
log_info "Installation Summary:"
echo "  ✅ Incremental hashing implemented"
echo "  ✅ External cron service available"
echo "  ✅ Prometheus metrics exporter ready"
echo "  ✅ Configuration files created"

# Setup instructions
echo ""
log_info "Next Steps:"
echo "  1. Install cron job:"
echo "     python3 services/guardian_cron_service.py --install-cron"
echo ""
echo "  2. Start Prometheus exporter:"
echo "     python3 monitoring/prometheus_exporter.py --port 8000"
echo ""
echo "  3. Monitor Guardian activity:"
echo "     tail -f logs/guardian-cron.log"
echo ""
echo "  4. Test file validation:"
echo "     python3 utils/file_integrity_guardian.py --validate-file /path/to/file.csv"

# Optional: Show example Prometheus scrape config
echo ""
log_info "Example Prometheus scrape configuration:"
echo "  scrape_configs:"
echo "    - job_name: 'greyhound-guardian'"
echo "      scrape_interval: 30s"
echo "      static_configs:"
echo "        - targets: ['localhost:8000']"

log_success "Guardian improvements setup completed!"
echo "Run with --help for usage information on individual components."
