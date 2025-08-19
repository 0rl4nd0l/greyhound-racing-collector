# Greyhound Racing Automation System

## Overview

This automation system provides complete hands-off operation of your greyhound racing data collection, analysis, and prediction system. It runs in the background on your Mac and automatically performs all necessary tasks on a schedule.

## Features

- **Automatic Data Collection**: Collects upcoming races from multiple venues
- **Live Odds Integration**: Updates live odds from Sportsbet throughout the day
- **Race Processing**: Processes historical race results automatically
- **ML Training & Backtesting**: Keeps prediction models up-to-date using Optuna Bayesian optimization
- **Advanced ML Pipeline**: Integrated SMOTE-NC, focal loss, and Platt calibration
- **MLflow Integration**: Automated experiment tracking and model registry
- **Report Generation**: Creates comprehensive analysis reports
- **Database Maintenance**: Automated backups and integrity checks
- **Value Bet Detection**: Identifies betting opportunities automatically

## Daily Schedule

- **07:00** - Morning routine (collect, odds, predict)
- **14:00** - Afternoon routine (process, analyze)
- **20:00** - Evening routine (ML, reports, maintenance)
- **Every 2 hours (9-22)** - Live odds updates
- **Sundays 22:00** - Weekly comprehensive ML training with Optuna optimization

## Installation

### 1. Install Required Dependencies

First, ensure you have the required Python packages:

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip3 install schedule selenium requests beautifulsoup4 pandas numpy scikit-learn

# Install ML optimization dependencies
pip3 install optuna mlflow imbalanced-learn seaborn
```

### 2. Install as macOS Service

To install the automation as a background service that starts automatically:

```bash
./automation_control.sh install
```

This will:
- Check all requirements
- Install the service to run automatically
- Start the automation immediately
- Configure it to start on system boot

## Usage

### Control Commands

```bash
# Check service status
./automation_control.sh status

# Start the service
./automation_control.sh start

# Stop the service
./automation_control.sh stop

# Restart the service
./automation_control.sh restart

# Uninstall the service
./automation_control.sh uninstall
```

### Manual Operation

Run automation manually in the foreground (useful for testing):

```bash
./automation_control.sh manual
```

### Single Task Execution

Run individual automation tasks for testing:

```bash
# Run morning routine
./automation_control.sh task morning

# Collect upcoming races
./automation_control.sh task collect

# Update live odds
./automation_control.sh task odds

# Run predictions
./automation_control.sh task predict

# Process historical races
./automation_control.sh task process

# Run ML backtesting
./automation_control.sh task ml

# Generate reports
./automation_control.sh task reports

# Backup database
./automation_control.sh task backup

# Data integrity check
./automation_control.sh task integrity

# Clean old files
./automation_control.sh task cleanup
```

### View Logs

```bash
# View automation logs
./automation_control.sh logs

# View system logs
./automation_control.sh logs launchd

# View error logs
./automation_control.sh logs error
```

## File Structure

The automation system creates the following structure:

```
greyhound_racing_collector/
├── logs/
│   └── automation/
│       ├── automation_YYYYMMDD.log      # Daily automation logs
│       ├── automation_status.json       # Current status
│       ├── launchd_out.log              # System output logs
│       └── launchd_err.log              # System error logs
├── backups/
│   └── greyhound_racing_data_backup_YYYYMMDD.db  # Daily database backups
├── automation_scheduler.py              # Main automation script
├── automation_control.sh                # Control script
└── com.greyhound.automation.plist       # macOS service configuration
```

## Monitoring

### Check Status

```bash
./automation_control.sh status
```

This shows:
- Whether the service is running
- Recent log entries
- Last successful operations
- Statistics and performance metrics

### Real-time Monitoring

Monitor logs in real-time:

```bash
tail -f logs/automation/automation_$(date +%Y%m%d).log
```

### Web Dashboard

Your Flask web dashboard continues to work normally and will show data collected by the automation system. Access it at:

```
http://localhost:5002
```

## Troubleshooting

### Service Won't Start

1. Check if Python 3 is installed: `python3 --version`
2. Verify dependencies: `pip3 list | grep -E "(schedule|selenium|requests|beautifulsoup4)"`
3. Check logs: `./automation_control.sh logs error`

### Tasks Failing

1. Check automation logs: `./automation_control.sh logs`
2. Run single task manually: `./automation_control.sh task collect`
3. Verify ChromeDriver is installed for odds collection

### Performance Issues

1. Monitor CPU usage: `top -pid $(pgrep -f automation_scheduler)`
2. Check disk space: `df -h`
3. Review log file sizes: `du -sh logs/`

### Database Issues

1. Check database integrity: `./automation_control.sh task integrity`
2. Restore from backup if needed: `cp backups/greyhound_racing_data_backup_YYYYMMDD.db greyhound_racing_data.db`

## Configuration

### Modify Schedule

Edit `automation_scheduler.py` to change the schedule:

```python
# Example: Change morning routine to 8:00 AM
schedule.every().day.at("08:00").do(self.morning_routine)
```

### Adjust Task Timeouts

Modify timeout values in `automation_scheduler.py`:

```python
# Example: Increase ML backtesting timeout to 30 minutes
timeout=1800  # 30 minutes in seconds
```

### Change Update Intervals

Modify odds update frequency:

```python
# Example: Update odds every hour instead of every 2 hours
for hour in range(9, 23, 1):  # Every hour from 9 AM to 10 PM
```

## Data Management

### Automatic Data Management

The system automatically:
- **Keeps ALL valuable data forever** (logs, predictions, race data, analysis results)
- **Database backups**: Smart retention (daily for 30 days, weekly for 12 weeks, monthly for 12 months, yearly forever)
- **Only removes**: Temporary files (*.tmp, *.pyc, __pycache__, .DS_Store, etc.)
- **Storage monitoring**: Tracks total usage and file counts for insight

### Manual Cleanup

```bash
# Clean old files immediately
./automation_control.sh task cleanup
```

### Storage Usage

Monitor storage usage:

```bash
# Check total project size
du -sh .

# Check database size
ls -lh *.db

# Check logs size
du -sh logs/
```

## Security

- The automation runs under your user account
- No network services are exposed
- All data remains on your local machine
- Database backups are stored locally
- Logs contain no sensitive information

## Performance Impact

The automation system is designed to be lightweight:
- Uses low CPU priority (nice value 10)
- Reduces activity during non-racing hours
- Implements efficient database operations
- Includes rate limiting for web scraping

Expected resource usage:
- **CPU**: 1-5% during active tasks, <1% idle
- **Memory**: 50-200MB depending on task
- **Disk**: ~30MB growth per day (data + logs)
- **Network**: Minimal, only for data collection

## Support

For issues or questions:

1. Check the logs first: `./automation_control.sh logs`
2. Try running tasks manually: `./automation_control.sh task [name]`
3. Restart the service: `./automation_control.sh restart`
4. Review this guide for configuration options

The automation system is designed to be self-healing and will continue running even if individual tasks fail occasionally.
