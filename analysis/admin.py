from django.contrib import admin
from .models import AnalysisJob, Claim, Evidence, UserFeedback

@admin.register(AnalysisJob)
class AnalysisJobAdmin(admin.ModelAdmin):
    list_display = ['id', 'content_type', 'status', 'verdict', 'confidence', 'user', 'created_at']
    list_filter = ['content_type', 'status', 'verdict', 'created_at']
    search_fields = ['id', 'text_content', 'user__username']
    readonly_fields = ['id', 'created_at', 'updated_at', 'completed_at']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('id', 'user', 'content_type', 'status')
        }),
        ('Input Data', {
            'fields': ('text_content', 'file_upload', 'url_input', 'metadata')
        }),
        ('Results', {
            'fields': ('verdict', 'confidence', 'summary', 'evidence', 'claims', 
                      'technical_appendix', 'recommended_action', 'human_review_required', 'human_steps')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'completed_at', 'processing_time')
        }),
        ('Errors', {
            'fields': ('error_message',)
        })
    )

@admin.register(Claim)
class ClaimAdmin(admin.ModelAdmin):
    list_display = ['job', 'claim_text', 'claim_verdict', 'claim_confidence', 'created_at']
    list_filter = ['claim_verdict', 'created_at']
    search_fields = ['claim_text', 'job__id']

@admin.register(Evidence)
class EvidenceAdmin(admin.ModelAdmin):
    list_display = ['job', 'evidence_type', 'method', 'score', 'created_at']
    list_filter = ['evidence_type', 'method', 'created_at']
    search_fields = ['job__id', 'method', 'explanation']

@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ['job', 'user', 'feedback_type', 'created_at']
    list_filter = ['feedback_type', 'created_at']
    search_fields = ['job__id', 'user__username', 'comment']
