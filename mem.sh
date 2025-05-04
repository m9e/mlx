#!/usr/bin/env bash

# Default values for percentages
DEFAULT_WIRED_LIMIT_PERCENT=98
DEFAULT_WIRED_LWM_PERCENT=90

# Read input parameters or use default values
WIRED_LIMIT_PERCENT=${1:-$DEFAULT_WIRED_LIMIT_PERCENT}
WIRED_LWM_PERCENT=${2:-$DEFAULT_WIRED_LWM_PERCENT}

# Validate inputs are within 0-100
if [[ $WIRED_LIMIT_PERCENT -lt 0 || $WIRED_LIMIT_PERCENT -gt 100 || $WIRED_LWM_PERCENT -lt 0 || $WIRED_LWM_PERCENT -gt 100 ]]; then
  echo "Error: Percentages must be between 0 and 100."
  exit 1
fi

# Get the total memory in MB
TOTAL_MEM_MB=$(($(sysctl -n hw.memsize) / 1024 / 1024))

# Calculate the memory limits
WIRED_LIMIT_MB=$(($TOTAL_MEM_MB * $WIRED_LIMIT_PERCENT / 100))
WIRED_LWM_MB=$(($TOTAL_MEM_MB * $WIRED_LWM_PERCENT / 100))

# Display the calculated values
echo "Total memory: $TOTAL_MEM_MB MB"
echo "Maximum limit (iogpu.wired_limit_mb): $WIRED_LIMIT_MB MB ($WIRED_LIMIT_PERCENT%)"
echo "Lower bound (iogpu.wired_lwm_mb): $WIRED_LWM_MB MB ($WIRED_LWM_PERCENT%)"

# Apply the values with sysctl
sudo sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB
sudo sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB
