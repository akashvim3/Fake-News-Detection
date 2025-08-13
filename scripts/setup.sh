#!/bin/bash

# Setup script for FactCheck API

echo "🚀 Setting up FactCheck API..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p media/uploads
mkdir -p static
mkdir -p models
mkdir -p backups

# Setup environment variables
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cat > .env << EOL
SECRET_KEY=django-insecure-change-me-in-production-$(openssl rand -hex 32)
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0
DB_NAME=factcheck_db
DB_USER=postgres
DB_PASSWORD=password
DB_HOST=localhost
DB_PORT=5432
REDIS_URL=redis://localhost:6379/0
BERT_MODEL_PATH=bert-base-uncased
DEEPFAKE_MODEL_PATH=models/deepfake_detector.pth
THROTTLE_ANON_RATE=100/hour
THROTTLE_USER_RATE=1000/hour
EOL
    echo "✅ .env file created. Please review and update as needed."
else
    echo "✅ .env file already exists"
fi

# Run migrations
echo "🗄️  Running database migrations..."
python manage.py makemigrations
python manage.py migrate

# Create superuser (optional)
echo "👤 Creating superuser..."
echo "from django.contrib.auth import get_user_model; User = get_user_model(); User.objects.create_superuser('admin', 'admin@factcheck.ai', 'admin123') if not User.objects.filter(username='admin').exists() else print('Superuser already exists')" | python manage.py shell

# Collect static files
echo "📦 Collecting static files..."
python manage.py collectstatic --noinput

# Test the setup
echo "🧪 Testing setup..."
python manage.py check

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Start Redis server: redis-server"
echo "2. Start Celery worker: celery -A factcheck_api worker --loglevel=info"
echo "3. Start Django server: python manage.py runserver"
echo ""
echo "🌐 Access points:"
echo "- API: http://localhost:8000/api/v1/"
echo "- Admin: http://localhost:8000/admin/ (admin/admin123)"
echo ""
echo "📖 For more information, see README.md"
