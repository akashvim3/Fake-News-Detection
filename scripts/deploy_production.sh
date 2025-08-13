#!/bin/bash

# Production deployment script

echo "Deploying FactCheck AI to production..."

# Set production environment
export DJANGO_SETTINGS_MODULE=factcheck_api.settings
export DEBUG=False

# Build Docker images
echo "Building Docker images..."
docker build -t factcheck-api:latest .
docker build -t factcheck-frontend:latest ./frontend/

# Tag images for registry
docker tag factcheck-api:latest your-registry.com/factcheck-api:latest
docker tag factcheck-frontend:latest your-registry.com/factcheck-frontend:latest

# Push to registry
echo "Pushing images to registry..."
docker push your-registry.com/factcheck-api:latest
docker push your-registry.com/factcheck-frontend:latest

# Deploy to Kubernetes
echo "Deploying to Kubernetes..."
kubectl apply -f kubernetes/

# Wait for deployment
echo "Waiting for deployment to complete..."
kubectl rollout status deployment/factcheck-api
kubectl rollout status deployment/celery-worker

# Run database migrations
echo "Running database migrations..."
kubectl exec -it deployment/factcheck-api -- python manage.py migrate

# Create superuser if needed
echo "Creating superuser..."
kubectl exec -it deployment/factcheck-api -- python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@factcheck.ai', 'secure_password_here')
    print('Superuser created')
else:
    print('Superuser already exists')
"

echo "✅ Production deployment completed!"
echo ""
echo "Services:"
echo "- API: https://api.factcheck.ai"
echo "- Frontend: https://factcheck.ai"
echo "- Admin: https://api.factcheck.ai/admin"
