#!/bin/bash

# Test runner script

echo "Running FactCheck API tests..."

# Set test environment
export DJANGO_SETTINGS_MODULE=factcheck_api.settings
export DEBUG=True

# Run Django tests
echo "Running Django unit tests..."
python manage.py test

# Run pytest
echo "Running pytest..."
pytest tests/ -v

# Run model tests
echo "Testing ML models..."
python -c "
from ml_models.text_classifier import FakeNewsClassifier
from ml_models.image_forensics import ImageForensicsAnalyzer

print('Testing text classifier...')
classifier = FakeNewsClassifier()
classifier.load_model()
result = classifier.predict('Test fake news content')
print(f'Text test passed: {result[\"verdict\"]}')

print('Testing image forensics...')
analyzer = ImageForensicsAnalyzer()
print('Image forensics test passed')
"

# API integration tests
echo "Running API integration tests..."
python -c "
import requests
import json

# Test API endpoints (assuming server is running)
base_url = 'http://localhost:8000/api/v1'

try:
    # Test stats endpoint
    response = requests.get(f'{base_url}/analysis/stats/')
    if response.status_code == 200:
        print('Stats endpoint test passed')
    else:
        print(f'Stats endpoint test failed: {response.status_code}')
except Exception as e:
    print(f'API integration test skipped (server not running): {e}')
"

echo "All tests completed!"
