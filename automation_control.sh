#!/bin/bash

# Greyhound Racing Automation Control Script
# Manages the automated data collection and analysis system

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_FILE="$SCRIPT_DIR/com.greyhound.automation.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.greyhound.automation.plist"
SERVICE_NAME="com.greyhound.automation"
PYTHON_SCRIPT="$SCRIPT_DIR/automation_scheduler.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    print_status "Checking requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check required Python packages
    python3 -c "import schedule, selenium, requests, beautifulsoup4" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "Some Python packages may be missing. Installing..."
        pip3 install schedule selenium requests beautifulsoup4 pandas numpy scikit-learn
    fi
    
    # Create logs directory
    mkdir -p "$SCRIPT_DIR/logs/automation"
    mkdir -p "$SCRIPT_DIR/backups"
    
    print_success "Requirements check completed"
}

install_service() {
    print_status "Installing automation service..."
    
    check_requirements
    
    # Copy plist file to LaunchAgents
    cp "$PLIST_FILE" "$PLIST_DEST"
    
    # Load the service
    launchctl load "$PLIST_DEST"
    
    if [ $? -eq 0 ]; then
        print_success "Automation service installed and started"
        print_status "Service will run in background and start automatically on boot"
    else
        print_error "Failed to install service"
        exit 1
    fi
}

uninstall_service() {
    print_status "Uninstalling automation service..."
    
    # Unload the service
    launchctl unload "$PLIST_DEST" 2>/dev/null
    
    # Remove plist file
    rm -f "$PLIST_DEST"
    
    print_success "Automation service uninstalled"
}

start_service() {
    print_status "Starting automation service..."
    
    if [ ! -f "$PLIST_DEST" ]; then
        print_error "Service not installed. Run: $0 install"
        exit 1
    fi
    
    launchctl load "$PLIST_DEST"
    print_success "Automation service started"
}

stop_service() {
    print_status "Stopping automation service..."
    
    launchctl unload "$PLIST_DEST" 2>/dev/null
    print_success "Automation service stopped"
}

restart_service() {
    print_status "Restarting automation service..."
    stop_service
    sleep 2
    start_service
}

status_service() {
    print_status "Checking automation service status..."
    
    # Check if service is loaded
    if launchctl list | grep -q "$SERVICE_NAME"; then
        print_success "Service is running"
        
        # Show recent log entries
        if [ -f "$SCRIPT_DIR/logs/automation/automation_$(date +%Y%m%d).log" ]; then
            print_status "Recent log entries:"
            tail -n 10 "$SCRIPT_DIR/logs/automation/automation_$(date +%Y%m%d).log"
        fi
        
        # Show status file if exists
        if [ -f "$SCRIPT_DIR/logs/automation/automation_status.json" ]; then
            print_status "Last status update:"
            cat "$SCRIPT_DIR/logs/automation/automation_status.json" | python3 -m json.tool
        fi
    else
        print_warning "Service is not running"
    fi
}

run_manual() {
    print_status "Running automation manually (Ctrl+C to stop)..."
    cd "$SCRIPT_DIR"
    python3 "$PYTHON_SCRIPT"
}

run_single_task() {
    if [ -z "$2" ]; then
        print_error "Task name required. Available tasks:"
        print_status "  morning    - Morning routine (collect, odds, predict)"
        print_status "  afternoon  - Afternoon routine (process, analyze)"
        print_status "  evening    - Evening routine (ML, reports, maintenance)"
        print_status "  collect    - Collect upcoming races"
        print_status "  process    - Process historical races"
        print_status "  odds       - Update Sportsbet odds"
        print_status "  predict    - Run race predictions"
        print_status "  ml         - Run ML backtesting"
        print_status "  reports    - Generate reports"
        print_status "  backup     - Backup database"
        print_status "  cleanup    - Clean old files"
        print_status "  integrity  - Data integrity check"
        exit 1
    fi
    
    print_status "Running single task: $2"
    cd "$SCRIPT_DIR"
    python3 "$PYTHON_SCRIPT" "$2"
}

show_logs() {
    LOG_TYPE=${2:-"automation"}
    
    case $LOG_TYPE in
        "automation")
            LOG_FILE="$SCRIPT_DIR/logs/automation/automation_$(date +%Y%m%d).log"
            ;;
        "launchd")
            LOG_FILE="$SCRIPT_DIR/logs/automation/launchd_out.log"
            ;;
        "error")
            LOG_FILE="$SCRIPT_DIR/logs/automation/launchd_err.log"
            ;;
        *)
            print_error "Unknown log type. Use: automation, launchd, or error"
            exit 1
            ;;
    esac
    
    if [ -f "$LOG_FILE" ]; then
        print_status "Showing logs from: $LOG_FILE"
        tail -n 50 "$LOG_FILE"
    else
        print_warning "Log file not found: $LOG_FILE"
    fi
}

show_help() {
    echo "Greyhound Racing Automation Control"
    echo "Usage: $0 {install|uninstall|start|stop|restart|status|manual|task|logs|help}"
    echo ""
    echo "Commands:"
    echo "  install     - Install automation service (runs automatically)"
    echo "  uninstall   - Remove automation service"
    echo "  start       - Start the service"
    echo "  stop        - Stop the service"
    echo "  restart     - Restart the service"
    echo "  status      - Show service status and recent activity"
    echo "  manual      - Run automation manually in foreground"
    echo "  task <name> - Run a single automation task"
    echo "  logs [type] - Show logs (automation|launchd|error)"
    echo "  help        - Show this help message"
    echo ""
    echo "The automation service runs these schedules:"
    echo "  • 07:00 - Morning routine (collect, odds, predict)"
    echo "  • 14:00 - Afternoon routine (process, analyze)"
    echo "  • 20:00 - Evening routine (ML, reports, maintenance)"
    echo "  • Every 2h (9-22) - Odds updates"
    echo "  • Sundays 22:00 - Weekly ML training"
}

# Main command handling
case "$1" in
    install)
        install_service
        ;;
    uninstall)
        uninstall_service
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    manual)
        run_manual
        ;;
    task)
        run_single_task "$@"
        ;;
    logs)
        show_logs "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
