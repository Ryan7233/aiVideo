#!/bin/bash

# Start Celery worker for AI Video processing

set -e

echo "Starting Celery worker..."

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Redis is not running. Starting Redis..."
    brew services start redis
    sleep 2
fi

# Kill any existing workers
pkill -f 'celery.*worker' || true
sleep 1

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Start Celery worker in background
echo "Starting Celery worker in background..."
nohup celery -A worker.celery_app worker \
    --loglevel=info \
    --concurrency=2 \
    --queues=default \
    --hostname=worker1@%h \
    > logs/celery_worker.out 2>&1 &

WORKER_PID=$!
echo "Celery worker started with PID: $WORKER_PID"

# Wait a moment and check if worker is running
sleep 3
if ps -p $WORKER_PID > /dev/null; then
    echo "✅ Celery worker is running successfully"
    echo "Logs: tail -f logs/celery_worker.out"
else
    echo "❌ Celery worker failed to start"
    echo "Check logs: cat logs/celery_worker.out"
    exit 1
fi
