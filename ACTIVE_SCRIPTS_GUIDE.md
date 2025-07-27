# ACTIVE SCRIPTS GUIDE
## Current Active Scripts for Greyhound Racing Prediction System

**Last Updated**: July 26, 2025
**For**: Future AI agents and developers

---

## 🎯 PRIMARY PREDICTION SCRIPTS (Current Active)

### 1. **weather_enhanced_predictor.py** - MAIN PREDICTOR
- **Status**: ✅ ACTIVE PRIMARY SCRIPT
- **Purpose**: Weather-enhanced ML predictor with comprehensive analysis
- **Used by**: Flask app (primary), automation system
- **Features**: Weather data integration, advanced ML models, comprehensive reporting
- **Command**: `python weather_enhanced_predictor.py [race_file.csv]`

### 2. **upcoming_race_predictor.py** - ORCHESTRATOR/FALLBACK
- **Status**: ✅ ACTIVE SECONDARY SCRIPT
- **Purpose**: Orchestrator wrapper script, fallback when weather predictor unavailable
- **Used by**: Flask app (fallback), automation system
- **Features**: Multi-race processing, wrapper functionality
- **Command**: `python upcoming_race_predictor.py [race_file.csv]`

### 3. **comprehensive_enhanced_ml_system.py** - ML CORE
- **Status**: ✅ ACTIVE CORE SYSTEM
- **Purpose**: Core ML system used internally by weather predictor
- **Used by**: weather_enhanced_predictor.py, ML training workflows
- **Features**: Advanced ML algorithms, feature engineering, model training
- **Command**: `python comprehensive_enhanced_ml_system.py --command analyze`

---

## 🚀 FLASK WEB APPLICATION

### **app.py** - MAIN WEB INTERFACE
- **Status**: ✅ ACTIVE WEB APP
- **Purpose**: Main Flask web application for race analysis dashboard
- **Port**: 5002
- **Features**: Race viewing, predictions, ML training, automation controls
- **URL**: http://localhost:5002

---

## 🤖 AUTOMATION & SCHEDULING

### **automation_scheduler.py** - AUTOMATION CORE
- **Status**: ✅ ACTIVE AUTOMATION
- **Purpose**: Automated daily tasks and scheduling
- **Features**: Data collection, predictions, backups, monitoring
- **Config**: com.greyhound.automation.plist (launchd service)

### **automation_control.sh** - SERVICE CONTROL
- **Status**: ✅ ACTIVE UTILITY
- **Purpose**: Start/stop/restart automation service
- **Commands**: `./automation_control.sh start|stop|restart|status`

---

## 📊 DATA PROCESSING & ANALYSIS

### **enhanced_race_analyzer.py** - ANALYTICS ENGINE
- **Status**: ✅ ACTIVE ANALYTICS
- **Purpose**: Advanced race data analysis and insights
- **Used by**: Flask app, analysis workflows

### **sportsbet_odds_integrator.py** - ODDS SYSTEM
- **Status**: ✅ ACTIVE ODDS
- **Purpose**: Live odds integration and value betting analysis
- **Used by**: Flask app, automation system

### **venue_mapping_fix.py** - VENUE MAPPER
- **Status**: ✅ ACTIVE UTILITY
- **Purpose**: Venue name standardization and mapping
- **Used by**: All prediction scripts

---

## 📁 DIRECTORY STRUCTURE (Active)

```
/greyhound_racing_collector/
├── 🎯 PREDICTION SCRIPTS
│   ├── weather_enhanced_predictor.py          # PRIMARY PREDICTOR
│   ├── upcoming_race_predictor.py             # FALLBACK/ORCHESTRATOR
│   └── comprehensive_enhanced_ml_system.py    # ML CORE
│
├── 🌐 WEB APPLICATION
│   ├── app.py                                 # MAIN FLASK APP
│   ├── templates/                             # HTML templates
│   └── static/                                # CSS/JS assets
│
├── 🤖 AUTOMATION
│   ├── automation_scheduler.py                # AUTOMATION CORE
│   ├── automation_control.sh                 # SERVICE CONTROL
│   └── com.greyhound.automation.plist        # LAUNCHD CONFIG
│
├── 📊 DATA & ANALYSIS
│   ├── enhanced_race_analyzer.py              # ANALYTICS
│   ├── sportsbet_odds_integrator.py           # ODDS SYSTEM
│   ├── venue_mapping_fix.py                  # VENUE MAPPING
│   └── greyhound_racing_data.db              # MAIN DATABASE
│
├── 📁 DATA DIRECTORIES
│   ├── upcoming_races/                       # Form guides for prediction
│   ├── predictions/                          # Generated predictions
│   ├── comprehensive_model_results/          # ML training results
│   └── logs/                                 # System logs
│
└── 📚 DOCUMENTATION
    ├── ACTIVE_SCRIPTS_GUIDE.md               # THIS FILE
    ├── AUTOMATION_GUIDE.md                   # Automation setup
    └── README.md                             # Project overview
```

---

## ⚡ QUICK START FOR NEW AGENTS

### To Run Predictions:
```bash
# Primary method (weather-enhanced)
python weather_enhanced_predictor.py path/to/race.csv

# Fallback method (if weather predictor unavailable)
python upcoming_race_predictor.py path/to/race.csv
```

### To Start Web Interface:
```bash
python app.py
# Then visit: http://localhost:5002
```

### To Control Automation:
```bash
./automation_control.sh start    # Start automation service
./automation_control.sh status   # Check service status
```

---

## 🚫 DEPRECATED/UNUSED FILES TO IGNORE

The following files/directories are outdated or for cleanup:
- `outdated_scripts/` - Old scripts (archived)
- `archive/` - Archived files
- `archive_unused_scripts/` - More archived scripts
- `cleanup_archive/` - Files moved during cleanup
- Various `.log` files - Old log files
- `Race_01_UNKNOWN_*.json` - Test/debug files
- Empty `.db` files - Unused databases
- `ml_env/`, `venv/` - Virtual environments

---

## 🔄 SCRIPT PRIORITY ORDER (Flask App)

### Background Predictions:
1. `weather_enhanced_predictor.py` (PRIMARY)
2. `upcoming_race_predictor.py` (FALLBACK)

### Single Race API:
1. `weather_enhanced_predictor.py` (PRIMARY)
2. `upcoming_race_predictor.py` (FALLBACK)

### ML Training:
1. `comprehensive_enhanced_ml_system.py` (CORE)

---

## 📝 NOTES FOR FUTURE AGENTS

1. **Always use `weather_enhanced_predictor.py` as the primary prediction script**
2. **Only fall back to `upcoming_race_predictor.py` if weather predictor fails**
3. **The Flask app (app.py) is the main user interface**
4. **All prediction results are stored in `./predictions/` directory**
5. **Main database is `greyhound_racing_data.db`**
6. **Automation runs via launchd service on macOS**

---

## 🆘 TROUBLESHOOTING

### If predictions fail:
1. Check if `weather_enhanced_predictor.py` exists and is executable
2. Fall back to `upcoming_race_predictor.py`
3. Ensure race CSV file is in correct format
4. Check logs in `./logs/` directory

### If web app fails:
1. Ensure port 5002 is available
2. Check Flask app logs
3. Verify database exists and is accessible

### If automation stops:
1. Run `./automation_control.sh status`
2. Check system logs: `tail -f logs/automation.log`
3. Restart service: `./automation_control.sh restart`

---

**Remember**: This system prioritizes weather-enhanced predictions while maintaining fallback capabilities for robust operation.
