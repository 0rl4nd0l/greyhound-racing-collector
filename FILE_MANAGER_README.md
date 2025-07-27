# Greyhound Racing File Manager

## Clear Data Visibility & Consistent Organization

This file management system provides **clear, consistent visibility** of all your data files across multiple interfaces.

## 📊 **EXACT CURRENT COUNTS**

Based on the latest scan:
- **Total Files: 6,832**
- **CSV Files: 188** (race data + analysis)
- **JSON Files: 6,644** (enhanced data + predictions)
- **Total Size: 168.7 MB**

### Breakdown by Category:
- **Race Data**: 6,653 files (44 CSV + 6,609 JSON)
- **ML Analysis**: 142 files (125 CSV + 17 JSON)
- **Other**: 36 files (18 CSV + 18 JSON)
- **Upcoming Races**: 1 file (1 CSV)

## 🚀 **Quick Start**

### Option 1: Simple Launcher
```bash
python launch_file_manager.py
```
This gives you a menu to choose:
1. Web UI (visual interface)
2. Command-line inventory (detailed report)
3. Quick file count
4. Exit

### Option 2: Direct Access

**Web UI (Visual Interface):**
```bash
streamlit run file_manager_ui.py
```

**Command-Line Inventory:**
```bash
python file_inventory.py
```

## 🎯 **Tools Overview**

### 1. **Streamlit Web UI** (`file_manager_ui.py`)
- **Visual, interactive interface**
- **5 tabs for different views:**
  - 📋 All Files - Complete file listing with filters
  - 🏁 Race Data - Race-specific files and track analysis
  - 🤖 ML & Analysis - Machine learning and analysis files
  - 📊 Data Explorer - Charts and visualizations
  - 🔍 File Search - Advanced search and filtering

**Features:**
- File preview and content inspection
- Interactive charts and graphs
- Filter by type, category, size, date
- Track coverage analysis
- Storage usage breakdown

### 2. **Command-Line Inventory** (`file_inventory.py`)
- **Precise, consistent file counts**
- **Comprehensive breakdown by:**
  - Category (Race Data, ML Analysis, etc.)
  - Directory (where files are located)
  - File type (CSV vs JSON)
  - Storage usage

**Features:**
- Exports detailed JSON report
- Verification commands provided
- Race data vs ML data breakdown
- Top directories by file count

### 3. **Simple Launcher** (`launch_file_manager.py`)
- **Easy menu-driven access**
- **Quick file counts**
- **No confusion about which tool to use**

## 📁 **File Categories Explained**

### Race Data (6,653 files)
- Raw race results
- Dog performance data
- Enhanced expert analysis
- **Location**: `enhanced_expert_data/`, `race_data/`, etc.

### ML Analysis (142 files)  
- Machine learning analysis per track
- Track-specific insights
- **Location**: `data/enhanced_data/`

### Predictions (73 files)
- ML-generated race predictions
- Betting recommendations with confidence
- **Location**: `predictions/`

### Other Categories
- Form Guides, Backtesting, Models, etc.

## 🔍 **Verification**

To manually verify file counts:
```bash
# CSV files
find . -name '*.csv' -not -path './backup_before_cleanup/*' -not -path './cleanup_archive/*' | wc -l

# JSON files  
find . -name '*.json' -not -path './backup_before_cleanup/*' -not -path './cleanup_archive/*' | wc -l

# Race CSV files specifically
find . -name 'Race_*.csv' -not -path './backup_before_cleanup/*' -not -path './cleanup_archive/*' | wc -l
```

## 🎨 **UI Features**

### Web Interface Highlights:
- **Real-time file scanning**
- **Interactive data tables**
- **File content preview**
- **Visual charts and graphs**
- **Advanced search capabilities**
- **Consistent data across all tabs**

### Command-Line Benefits:
- **Exact, verifiable counts**
- **Fast execution**
- **Detailed JSON export**
- **Perfect for automation**

## 🛠 **Requirements**

For Web UI:
```bash
pip install streamlit plotly pandas
```

For Command-Line:
```bash
# Only standard Python libraries needed
```

## 🎯 **Why This Solves the Confusion**

1. **Consistent Exclusions**: All tools exclude the same directories (backups, archives)
2. **Same Categorization**: Files are categorized consistently across all interfaces
3. **Exact Counts**: No more discrepancies - same scanning logic everywhere
4. **Multiple Views**: Choose the interface that works best for your needs
5. **Verification**: Commands provided to manually verify all counts

## 📈 **Usage Recommendations**

- **Daily Use**: Web UI for interactive exploration
- **Reports**: Command-line tool for exact counts
- **Quick Checks**: Launcher menu option 3
- **Analysis**: Web UI tabs 3 & 4 for insights
- **Search**: Web UI tab 5 for finding specific files

Your **6,832 files** are now clearly organized and easily accessible!
