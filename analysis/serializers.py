from rest_framework import serializers
from .models import AnalysisJob, UserFeedback, Claim, Evidence

class EvidenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Evidence
        fields = ['evidence_type', 'method', 'score', 'explanation', 'metadata']

class ClaimSerializer(serializers.ModelSerializer):
    class Meta:
        model = Claim
        fields = ['claim_text', 'claim_verdict', 'claim_confidence', 'sources']

class AnalysisJobSerializer(serializers.ModelSerializer):
    evidence_items = EvidenceSerializer(many=True, read_only=True)
    extracted_claims = ClaimSerializer(many=True, read_only=True)
    
    class Meta:
        model = AnalysisJob
        fields = [
            'id', 'content_type', 'status', 'verdict', 'confidence',
            'summary', 'evidence', 'claims', 'technical_appendix',
            'recommended_action', 'human_review_required', 'human_steps',
            'created_at', 'updated_at', 'completed_at', 'processing_time',
            'evidence_items', 'extracted_claims'
        ]
        read_only_fields = [
            'id', 'status', 'verdict', 'confidence', 'summary',
            'evidence', 'claims', 'technical_appendix', 'recommended_action',
            'human_review_required', 'human_steps', 'created_at',
            'updated_at', 'completed_at', 'processing_time'
        ]

class AnalysisRequestSerializer(serializers.Serializer):
    content_type = serializers.ChoiceField(choices=[
        ('text', 'Text'),
        ('image', 'Image'),
        ('video', 'Video'),
        ('audio', 'Audio'),
        ('multimodal', 'Multimodal'),
    ])
    text_content = serializers.CharField(required=False, allow_blank=True)
    file_upload = serializers.FileField(required=False)
    url_input = serializers.URLField(required=False, allow_blank=True)
    metadata = serializers.JSONField(required=False, default=dict)
    
    def validate(self, data):
        content_type = data.get('content_type')
        text_content = data.get('text_content')
        file_upload = data.get('file_upload')
        url_input = data.get('url_input')
        
        if content_type == 'text' and not text_content:
            raise serializers.ValidationError("Text content is required for text analysis")
        
        if content_type in ['image', 'video', 'audio'] and not file_upload and not url_input:
            raise serializers.ValidationError("File upload or URL is required for media analysis")
        
        return data

class UserFeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserFeedback
        fields = ['feedback_type', 'comment']
