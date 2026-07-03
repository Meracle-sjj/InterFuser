#!/bin/bash
# Monitor training progress

# Check if training process is running
if pgrep -f "train.py" > /dev/null; then
    echo "✓ Training process is running"
    echo ""
    
    # Show GPU usage
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader | sed 's/,/ | /g'
    echo ""
    
    # Find the training log file
    LATEST_LOG=$(find /home/shijj/interfuser/interfuser -name "*.log" -type f 2>/dev/null | sort | tail -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "=== Latest Training Log: $LATEST_LOG ==="
        tail -30 "$LATEST_LOG"
    else
        echo "=== Checking for recent output directories ==="
        OUTPUT_DIR=$(ls -td /home/shijj/interfuser/interfuser/output/* 2>/dev/null | head -1)
        if [ -n "$OUTPUT_DIR" ]; then
            echo "Latest output directory: $OUTPUT_DIR"
            ls -la "$OUTPUT_DIR" | tail -10
        fi
    fi
else
    echo "✗ No training process running"
fi
