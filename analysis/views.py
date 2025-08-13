from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.shortcuts import get_object_or_404
from django.utils import timezone
from .models import AnalysisJob, UserFeedback
from .serializers import (
    AnalysisJobSerializer, 
    AnalysisRequestSerializer, 
    UserFeedbackSerializer
)
from .tasks import process_analysis_job
import logging

logger = logging.getLogger(__name__)

class AnalysisViewSet(viewsets.ModelViewSet):
    queryset = AnalysisJob.objects.all()
    serializer_class = AnalysisJobSerializer
    permission_classes = [AllowAny]  # Change to IsAuthenticated in production
    
    def get_queryset(self):
        if self.request.user.is_authenticated:
            return AnalysisJob.objects.filter(user=self.request.user)
        return AnalysisJob.objects.none()
    
    @action(detail=False, methods=['post'], url_path='analyze')
    def analyze(self, request):
        """Submit content for analysis"""
        serializer = AnalysisRequestSerializer(data=request.data)
        if serializer.is_valid():
            # Create analysis job
            job_data = serializer.validated_data
            job = AnalysisJob.objects.create(
                user=request.user if request.user.is_authenticated else None,
                content_type=job_data['content_type'],
                text_content=job_data.get('text_content'),
                file_upload=job_data.get('file_upload'),
                url_input=job_data.get('url_input'),
                metadata=job_data.get('metadata', {}),
                status='pending'
            )
            
            # Queue processing task
            try:
                process_analysis_job.delay(str(job.id))
                logger.info(f"Queued analysis job {job.id}")
            except Exception as e:
                logger.error(f"Failed to queue job {job.id}: {str(e)}")
                job.status = 'failed'
                job.error_message = f"Failed to queue job: {str(e)}"
                job.save()
            
            return Response({
                'job_id': job.id,
                'status': job.status,
                'message': 'Analysis job submitted successfully'
            }, status=status.HTTP_202_ACCEPTED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['get'], url_path='status')
    def get_status(self, request, pk=None):
        """Get analysis job status and results"""
        job = get_object_or_404(AnalysisJob, pk=pk)
        
        # Check if user has permission to view this job
        if job.user and job.user != request.user and not request.user.is_staff:
            return Response(
                {'error': 'Permission denied'}, 
                status=status.HTTP_403_FORBIDDEN
            )
        
        serializer = AnalysisJobSerializer(job)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'], url_path='feedback')
    def submit_feedback(self, request, pk=None):
        """Submit user feedback on analysis results"""
        job = get_object_or_404(AnalysisJob, pk=pk)
        
        if not request.user.is_authenticated:
            return Response(
                {'error': 'Authentication required'}, 
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        serializer = UserFeedbackSerializer(data=request.data)
        if serializer.is_valid():
            feedback, created = UserFeedback.objects.update_or_create(
                job=job,
                user=request.user,
                defaults=serializer.validated_data
            )
            
            return Response({
                'message': 'Feedback submitted successfully',
                'created': created
            }, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['get'], url_path='stats')
    def get_stats(self, request):
        """Get analysis statistics"""
        if request.user.is_authenticated:
            user_jobs = AnalysisJob.objects.filter(user=request.user)
        else:
            user_jobs = AnalysisJob.objects.none()
        
        stats = {
            'total_analyses': user_jobs.count(),
            'completed': user_jobs.filter(status='completed').count(),
            'pending': user_jobs.filter(status='pending').count(),
            'processing': user_jobs.filter(status='processing').count(),
            'failed': user_jobs.filter(status='failed').count(),
            'by_content_type': {
                'text': user_jobs.filter(content_type='text').count(),
                'image': user_jobs.filter(content_type='image').count(),
                'video': user_jobs.filter(content_type='video').count(),
                'audio': user_jobs.filter(content_type='audio').count(),
            }
        }
        
        return Response(stats)
