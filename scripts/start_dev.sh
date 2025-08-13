#!/bin/bash

# Development startup script

echo "🚀 Starting FactCheck AI development environment..."

# Check if setup has been run
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Running setup first..."
    ./scripts/setup.sh
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Running setup first..."
    ./scripts/setup.sh
fi

# Check if Redis is running
if ! pgrep -x "redis-server" > /dev/null; then
    echo "🔴 Starting Redis server..."
    redis-server --daemonize yes --logfile logs/redis.log
    sleep 2
fi

# Run migrations
echo "🗄️  Running migrations..."
python manage.py migrate

# Start Celery worker in background
echo "🌿 Starting Celery worker..."
celery -A factcheck_api worker --loglevel=info --logfile=logs/celery.log --detach

# Start Django development server
echo "🌐 Starting Django development server..."
echo ""
echo "✅ Development environment started!"
echo ""
echo "🌐 Access points:"
echo "- API: http://localhost:8000/api/v1/"
echo "- Admin: http://localhost:8000/admin/ (admin/admin123)"
echo "- Health Check: http://localhost:8000/api/v1/analysis/stats/"
echo ""
echo "📊 To monitor:"
echo "- Celery logs: tail -f logs/celery.log"
echo "- Redis logs: tail -f logs/redis.log"
echo ""
echo "🛑 To stop: ./scripts/stop_dev.sh"
echo ""

python manage.py runserver 0.0.0.0:8000
