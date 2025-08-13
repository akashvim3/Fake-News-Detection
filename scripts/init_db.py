#!/usr/bin/env python3
"""
Database initialization script
"""
import os
import sys
import django
from django.core.management import execute_from_command_line
from django.contrib.auth import get_user_model

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'factcheck_api.settings')
django.setup()

def create_superuser():
    """Create superuser if it doesn't exist"""
    User = get_user_model()
    
    if not User.objects.filter(username='admin').exists():
        print("Creating superuser...")
        User.objects.create_superuser(
            username='admin',
            email='admin@factcheck.ai',
            password='admin123',
            organization='FactCheck AI'
        )
        print("Superuser created: admin/admin123")
    else:
        print("Superuser already exists")

def create_sample_data():
    """Create sample analysis jobs for testing"""
    from analysis.models import AnalysisJob
    from django.contrib.auth import get_user_model
    
    User = get_user_model()
    admin_user = User.objects.get(username='admin')
    
    sample_jobs = [
        {
            'content_type': 'text',
            'text_content': 'Breaking: Scientists discover that vaccines contain microchips for government surveillance.',
            'status': 'completed',
            'verdict': 'likely_misinformation',
            'confidence': 0.92,
            'summary': 'Analysis indicates high probability of misinformation based on language patterns and fact-checking databases.'
        },
        {
            'content_type': 'text',
            'text_content': 'New study shows that regular exercise can improve mental health and reduce anxiety.',
            'status': 'completed',
            'verdict': 'likely_true',
            'confidence': 0.88,
            'summary': 'Content appears to be factual and consistent with established scientific research.'
        },
        {
            'content_type': 'image',
            'status': 'completed',
            'verdict': 'likely_deepfake',
            'confidence': 0.76,
            'summary': 'Image forensics detected potential manipulation artifacts.'
        }
    ]
    
    for job_data in sample_jobs:
        if not AnalysisJob.objects.filter(text_content=job_data.get('text_content')).exists():
            AnalysisJob.objects.create(
                user=admin_user,
                **job_data
            )
    
    print(f"Created {len(sample_jobs)} sample analysis jobs")

def main():
    """Main initialization function"""
    print("Initializing FactCheck AI database...")
    
    # Run migrations
    print("Running migrations...")
    execute_from_command_line(['manage.py', 'migrate'])
    
    # Create superuser
    create_superuser()
    
    # Create sample data
    create_sample_data()
    
    print("Database initialization completed!")
    print("\nYou can now:")
    print("1. Start the development server: python manage.py runserver")
    print("2. Access admin panel: http://localhost:8000/admin (admin/admin123)")
    print("3. Start Celery worker: celery -A factcheck_api worker --loglevel=info")

if __name__ == '__main__':
    main()
