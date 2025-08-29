#!/bin/bash

# Start Flower monitoring for Celery tasks

set -e

echo "Starting Flower monitoring..."

# Kill any existing Flower processes
pkill -f 'flower' || true
sleep 1

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Start Flower in background
echo "Starting Flower on http://localhost:5555"
nohup celery -A worker.celery_app flower \
    --port=5555 \
    --broker=redis://127.0.0.1:6379/0 \
    --basic_auth=admin:admin123 \
    > logs/flower.out 2>&1 &

FLOWER_PID=$!
echo "Flower started with PID: $FLOWER_PID"

# Wait a moment and check if Flower is running
sleep 3
if ps -p $FLOWER_PID > /dev/null; then
    echo "✅ Flower is running successfully"
    echo "🌸 Open http://localhost:5555 (admin/admin123)"
    echo "Logs: tail -f logs/flower.out"
else
    echo "❌ Flower failed to start"
    echo "Check logs: cat logs/flower.out"
    exit 1
fi
