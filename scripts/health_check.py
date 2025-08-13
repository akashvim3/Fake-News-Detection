#!/usr/bin/env python3
"""
Health check script for FactCheck API
"""
import requests
import sys
import json
import time
import os
import django
from datetime import datetime

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'factcheck_api.settings')
django.setup()

def check_api_health(base_url="http://localhost:8000"):
    """Check API health endpoints"""
    endpoints = [
        "/api/v1/analysis/stats/",
        "/admin/login/",  # Changed from /admin/ to avoid redirect
    ]
    
    results = {}
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10, allow_redirects=False)
            response_time = time.time() - start_time
            
            # Consider 200-399 as healthy
            is_healthy = 200 <= response.status_code < 400
            
            results[endpoint] = {
                "status": "healthy" if is_healthy else "unhealthy",
                "status_code": response.status_code,
                "response_time": round(response_time * 1000, 2),  # ms
                "timestamp": datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            results[endpoint] = {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    return results

def check_database():
    """Check database connectivity"""
    try:
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            return {"status": "healthy", "message": "Database connection successful"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

def check_redis():
    """Check Redis connectivity"""
    try:
        import redis
        from django.conf import settings
        
        # Parse Redis URL
        redis_url = getattr(settings, 'CELERY_BROKER_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        r.ping()
        return {"status": "healthy", "message": "Redis connection successful"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

def check_celery():
    """Check Celery worker status"""
    try:
        from celery import current_app
        
        # Get active workers
        inspect = current_app.control.inspect(timeout=5)
        if inspect is None:
            return {"status": "unhealthy", "message": "Cannot connect to Celery"}
            
        active_workers = inspect.active()
        
        if active_workers:
            return {
                "status": "healthy", 
                "workers": len(active_workers),
                "worker_names": list(active_workers.keys())
            }
        else:
            return {"status": "unhealthy", "message": "No active Celery workers"}
            
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

def check_file_permissions():
    """Check file system permissions"""
    try:
        from django.conf import settings
        
        # Check media directory
        media_root = getattr(settings, 'MEDIA_ROOT', 'media')
        if not os.path.exists(media_root):
            os.makedirs(media_root, exist_ok=True)
        
        # Test write permission
        test_file = os.path.join(media_root, 'health_check_test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
        return {"status": "healthy", "message": "File system permissions OK"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

def main():
    """Main health check function"""
    print("🏥 FactCheck AI Health Check")
    print("=" * 40)
    
    all_healthy = True
    
    # Check API endpoints
    print("\n📡 API Endpoints:")
    api_results = check_api_health()
    for endpoint, result in api_results.items():
        status_icon = "✅" if result["status"] == "healthy" else "❌"
        print(f"{status_icon} {endpoint}: {result['status']}")
        if "response_time" in result:
            print(f"   Response time: {result['response_time']}ms")
        if "error" in result:
            print(f"   Error: {result['error']}")
            all_healthy = False
    
    # Check database
    print("\n🗄️  Database:")
    db_result = check_database()
    status_icon = "✅" if db_result["status"] == "healthy" else "❌"
    print(f"{status_icon} Database: {db_result['status']}")
    if "error" in db_result:
        print(f"   Error: {db_result['error']}")
        all_healthy = False
    
    # Check Redis
    print("\n🔴 Redis:")
    redis_result = check_redis()
    status_icon = "✅" if redis_result["status"] == "healthy" else "❌"
    print(f"{status_icon} Redis: {redis_result['status']}")
    if "error" in redis_result:
        print(f"   Error: {redis_result['error']}")
        all_healthy = False
    
    # Check Celery
    print("\n🌿 Celery:")
    celery_result = check_celery()
    status_icon = "✅" if celery_result["status"] == "healthy" else "❌"
    print(f"{status_icon} Celery Workers: {celery_result['status']}")
    if "workers" in celery_result:
        print(f"   Active workers: {celery_result['workers']}")
    if "error" in celery_result:
        print(f"   Error: {celery_result['error']}")
        all_healthy = False
    
    # Check file permissions
    print("\n📁 File System:")
    fs_result = check_file_permissions()
    status_icon = "✅" if fs_result["status"] == "healthy" else "❌"
    print(f"{status_icon} File Permissions: {fs_result['status']}")
    if "error" in fs_result:
        print(f"   Error: {fs_result['error']}")
        all_healthy = False
    
    # Overall status
    print("\n" + "=" * 40)
    if all_healthy:
        print("🎉 All systems healthy!")
        sys.exit(0)
    else:
        print("⚠️  Some issues detected")
        sys.exit(1)

if __name__ == "__main__":
    main()
