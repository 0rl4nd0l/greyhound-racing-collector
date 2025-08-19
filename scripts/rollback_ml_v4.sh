#!/bin/bash
# Rollback ML System V4 changes

set -euo pipefail

echo "Rolling back ML System V4..."

# Timestamped backup dir
TS=$(date +%Y%m%d_%H%M%S)
BK_DIR="backups/v4_${TS}"

# Ensure backup dir exists
mkdir -p "$BK_DIR"

# Backup current V4 files (non-failing if none match)
shopt -s nullglob
v4_files=(*v4*.py)
if [ ${#v4_files[@]} -gt 0 ]; then
  cp "${v4_files[@]}" "$BK_DIR"/
  echo "Backed up ${#v4_files[@]} file(s) to $BK_DIR"
else
  echo "No *v4*.py files found to back up"
fi
shopt -u nullglob

# Restore previous version from archive if available
if [ -f "archive/ml_system_v3.py" ]; then
    cp archive/ml_system_v3.py ml_system.py
    echo "Restored V3 as fallback"
else
    echo "No V3 fallback found at archive/ml_system_v3.py"
fi

# Restart services (uncomment as appropriate)
# systemctl restart ml-api  # If using systemd
# pm2 restart ml-api        # If using pm2

echo "Rollback complete. Please verify system functionality."
