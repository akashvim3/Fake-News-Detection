from django.db import models
from django.contrib.auth.models import User
import uuid

class AnalysisJob(models.Model):
    CONTENT_TYPES = [
        ('text', 'Text'),
        ('image', 'Image'),
        ('video', 'Video'),
        ('audio', 'Audio'),
        ('multimodal', 'Multimodal'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    VERDICT_CHOICES = [
        ('likely_misinformation', 'Likely Misinformation'),
        ('likely_true', 'Likely True'),
        ('inconclusive', 'Inconclusive'),
        ('likely_deepfake', 'Likely Deepfake'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    content_type = models.CharField(max_length=20, choices=CONTENT_TYPES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Input data
    text_content = models.TextField(blank=True, null=True)
    file_upload = models.FileField(upload_to='uploads/', blank=True, null=True)
    url_input = models.URLField(blank=True, null=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    # Results
    verdict = models.CharField(max_length=30, choices=VERDICT_CHOICES, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    summary = models.TextField(blank=True, null=True)
    evidence = models.JSONField(default=list, blank=True)
    claims = models.JSONField(default=list, blank=True)
    technical_appendix = models.JSONField(default=dict, blank=True)
    recommended_action = models.CharField(max_length=50, blank=True, null=True)
    human_review_required = models.BooleanField(default=False)
    human_steps = models.JSONField(default=list, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    
    # Processing info
    processing_time = models.FloatField(blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.content_type} analysis - {self.status}"

class Claim(models.Model):
    job = models.ForeignKey(AnalysisJob, on_delete=models.CASCADE, related_name='extracted_claims')
    claim_text = models.TextField()
    claim_verdict = models.CharField(max_length=20, choices=[
        ('true', 'True'),
        ('false', 'False'),
        ('inconclusive', 'Inconclusive'),
    ])
    claim_confidence = models.FloatField()
    sources = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

class Evidence(models.Model):
    EVIDENCE_TYPES = [
        ('source', 'Source'),
        ('forensic', 'Forensic'),
        ('metadata', 'Metadata'),
    ]
    
    job = models.ForeignKey(AnalysisJob, on_delete=models.CASCADE, related_name='evidence_items')
    evidence_type = models.CharField(max_length=20, choices=EVIDENCE_TYPES)
    method = models.CharField(max_length=100)
    score = models.FloatField()
    explanation = models.TextField()
    metadata = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

class UserFeedback(models.Model):
    FEEDBACK_TYPES = [
        ('agree', 'Agree'),
        ('disagree', 'Disagree'),
        ('false_positive', 'False Positive'),
        ('false_negative', 'False Negative'),
    ]
    
    job = models.ForeignKey(AnalysisJob, on_delete=models.CASCADE, related_name='feedback')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_TYPES)
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['job', 'user']
