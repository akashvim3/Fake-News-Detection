#!/usr/bin/env python3
"""
Database backup script
"""
import os
import sys
import django
from datetime import datetime
import subprocess
import json

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'factcheck_api.settings')
django.setup()

from django.conf import settings
from django.core.management import call_command
from django.contrib.auth import get_user_model
from analysis.models import AnalysisJob

def backup_database():
    """Create database backup"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = 'backups'
    os.makedirs(backup_dir, exist_ok=True)
    
    # Django fixture backup
    fixture_file = f"{backup_dir}/factcheck_backup_{timestamp}.json"
    
    print(f"Creating Django fixture backup: {fixture_file}")
    with open(fixture_file, 'w') as f:
        call_command('dumpdata', stdout=f, indent=2)
    
    # PostgreSQL dump (if using PostgreSQL)
    if 'postgresql' in settings.DATABASES['default']['ENGINE']:
        pg_dump_file = f"{backup_dir}/factcheck_pg_dump_{timestamp}.sql"
        
        db_config = settings.DATABASES['default']
        pg_dump_cmd = [
            'pg_dump',
            f"--host={db_config['HOST']}",
            f"--port={db_config['PORT']}",
            f"--username={db_config['USER']}",
            f"--dbname={db_config['NAME']}",
            f"--file={pg_dump_file}",
            '--verbose'
        ]
        
        print(f"Creating PostgreSQL dump: {pg_dump_file}")
        try:
            subprocess.run(pg_dump_cmd, check=True, 
                         env={**os.environ, 'PGPASSWORD': db_config['PASSWORD']})
        except subprocess.CalledProcessError as e:
            print(f"PostgreSQL dump failed: {e}")
    
    return fixture_file

def create_backup_info():
    """Create backup information file"""
    User = get_user_model()
    
    info = {
        'backup_date': datetime.now().isoformat(),
        'django_version': django.get_version(),
        'database_engine': settings.DATABASES['default']['ENGINE'],
        'statistics': {
            'total_users': User.objects.count(),
            'total_analyses': AnalysisJob.objects.count(),
            'completed_analyses': AnalysisJob.objects.filter(status='completed').count(),
            'pending_analyses': AnalysisJob.objects.filter(status='pending').count(),
        }
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    info_file = f"backups/backup_info_{timestamp}.json"
    
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Backup info saved: {info_file}")
    return info_file

def main():
    """Main backup function"""
    print("Starting FactCheck AI database backup...")
    
    try:
        # Create backup
        backup_file = backup_database()
        
        # Create backup info
        info_file = create_backup_info()
        
        print("\n✅ Backup completed successfully!")
        print(f"Backup file: {backup_file}")
        print(f"Info file: {info_file}")
        
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
