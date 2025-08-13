#!/bin/bash

# Comprehensive system check script

echo "🔍 Running comprehensive FactCheck AI system checks..."
echo "=================================================="

# Create logs directory
mkdir -p logs

# 1. Health Check
echo ""
echo "1️⃣  Running health check..."
python scripts/health_check.py | tee logs/health_check.log

# 2. Performance Test
echo ""
echo "2️⃣  Running performance tests..."
python scripts/performance_test.py | tee logs/performance_test.log

# 3. Model Evaluation
echo ""
echo "3️⃣  Running model evaluation..."
python scripts/model_evaluation.py | tee logs/model_evaluation.log

# 4. Database Backup
echo ""
echo "4️⃣  Creating database backup..."
python scripts/backup_db.py | tee logs/backup.log

# 5. Run Django Tests
echo ""
echo "5️⃣  Running Django unit tests..."
python manage.py test | tee logs/django_tests.log

# 6. Security Check
echo ""
echo "6️⃣  Running security checks..."
python manage.py check --deploy | tee logs/security_check.log

# Summary
echo ""
echo "=================================================="
echo "✅ All system checks completed!"
echo ""
echo "📁 Log files saved in logs/ directory:"
echo "   - health_check.log"
echo "   - performance_test.log"
echo "   - model_evaluation.log"
echo "   - backup.log"
echo "   - django_tests.log"
echo "   - security_check.log"
echo ""
echo "📊 Check individual log files for detailed results"
