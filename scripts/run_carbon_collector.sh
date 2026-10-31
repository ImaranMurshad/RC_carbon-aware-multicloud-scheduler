
#!/bin/bash

# ==========================================
# CARBON-AWARE MULTI-CLOUD SCHEDULER
# Automated Carbon Data Collection
# ==========================================

# Project directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Python and Django paths
PYTHON="$PROJECT_ROOT/.venv/bin/python"
MANAGE="$PROJECT_ROOT/backend/manage.py"

# Log configuration
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/carbon_collector.log"

# Collection interval: 15 minutes (900 seconds)
INTERVAL=900
INTERVAL_MINUTES=$((INTERVAL / 60))

# Create log directory if it does not exist
mkdir -p "$LOG_DIR"

# Check required files
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python virtual environment not found: $PYTHON"
    exit 1
fi

if [ ! -f "$MANAGE" ]; then
    echo "ERROR: Django manage.py not found: $MANAGE"
    exit 1
fi

# Start logging
echo "==========================================" >> "$LOG_FILE"
echo "Carbon Collector Started: $(date)" >> "$LOG_FILE"
echo "Collection interval: ${INTERVAL_MINUTES} minutes" >> "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

# Handle graceful shutdown
cleanup() {
    echo "" >> "$LOG_FILE"
    echo "Carbon Collector stopped: $(date)" >> "$LOG_FILE"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Continuous collection
while true
do
    echo "" >> "$LOG_FILE"
    echo "------------------------------------------" >> "$LOG_FILE"
    echo "Collection started: $(date)" >> "$LOG_FILE"

    "$PYTHON" "$MANAGE" collect_carbon >> "$LOG_FILE" 2>&1

    EXIT_CODE=$?

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "Collection completed successfully: $(date)" >> "$LOG_FILE"
    else
        echo "Collection failed with exit code $EXIT_CODE: $(date)" >> "$LOG_FILE"
    fi

    echo "Next collection in ${INTERVAL_MINUTES} minutes." >> "$LOG_FILE"

    sleep "$INTERVAL"
done
