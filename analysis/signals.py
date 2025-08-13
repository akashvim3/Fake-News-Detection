"""
Django signals for analysis app
"""
from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver
from django.core.cache import cache
from .models import AnalysisJob, UserFeedback
import logging

logger = logging.getLogger(__name__)

@receiver(post_save, sender=AnalysisJob)
def analysis_job_saved(sender, instance, created, **kwargs):
    """Handle analysis job save events"""
    if created:
        logger.info(f"New analysis job created: {instance.id}")
        
        # Clear user stats cache
        if instance.user:
            cache_key = f"user_stats_{instance.user.id}"
            cache.delete(cache_key)
    
    elif instance.status == 'completed':
        logger.info(f"Analysis job completed: {instance.id}")
        
        # Update user stats cache
        if instance.user:
            cache_key = f"user_stats_{instance.user.id}"
            cache.delete(cache_key)

@receiver(post_save, sender=UserFeedback)
def user_feedback_saved(sender, instance, created, **kwargs):
    """Handle user feedback events"""
    if created:
        logger.info(f"New feedback received for job {instance.job.id}")
        
        # Could trigger model retraining pipeline here
        # retrain_model.delay(instance.feedback_type, instance.job.id)

@receiver(pre_delete, sender=AnalysisJob)
def analysis_job_deleted(sender, instance, **kwargs):
    """Handle analysis job deletion"""
    logger.info(f"Analysis job deleted: {instance.id}")
    
    # Clean up associated files
    if instance.file_upload:
        try:
            instance.file_upload.delete(save=False)
        except Exception as e:
            logger.error(f"Failed to delete file for job {instance.id}: {e}")
