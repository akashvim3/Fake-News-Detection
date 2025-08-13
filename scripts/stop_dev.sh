#!/bin/bash

# Development shutdown script

echo "🛑 Stopping FactCheck AI development environment..."

# Stop Django server
echo "🌐 Stopping Django server..."
pkill -f "python manage.py runserver" 2>/dev/null || echo "Django server not running"

# Stop Celery worker
echo "🌿 Stopping Celery worker..."
pkill -f "celery.*worker" 2>/dev/null || echo "Celery worker not running"

# Stop Redis server (optional - comment out if you want to keep Redis running)
echo "🔴 Stopping Redis server..."
redis-cli shutdown 2>/dev/null || echo "Redis server not running"

echo "✅ Development environment stopped!"
