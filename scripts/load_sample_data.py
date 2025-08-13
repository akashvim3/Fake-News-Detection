#!/usr/bin/env python3
"""
Load sample data for testing and demonstration
"""
import os
import sys
import django
import json
from datetime import datetime, timedelta

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'factcheck_api.settings')
django.setup()

from django.contrib.auth import get_user_model
from analysis.models import AnalysisJob, Claim, Evidence, UserFeedback

User = get_user_model()

def create_test_users():
    """Create test users"""
    users_data = [
        {
            'username': 'journalist1',
            'email': 'journalist@news.com',
            'password': 'testpass123',
            'first_name': 'Jane',
            'last_name': 'Reporter',
            'organization': 'News Corp'
        },
        {
            'username': 'researcher1',
            'email': 'researcher@university.edu',
            'password': 'testpass123',
            'first_name': 'Dr. John',
            'last_name': 'Smith',
            'organization': 'University Research Lab'
        },
        {
            'username': 'factchecker1',
            'email': 'checker@factcheck.org',
            'password': 'testpass123',
            'first_name': 'Maria',
            'last_name': 'Validator',
            'organization': 'FactCheck.org'
        }
    ]
    
    created_users = []
    for user_data in users_data:
        user, created = User.objects.get_or_create(
            username=user_data['username'],
            defaults=user_data
        )
        if created:
            user.set_password(user_data['password'])
            user.save()
            created_users.append(user)
    
    print(f"Created {len(created_users)} test users")
    return User.objects.filter(username__in=[u['username'] for u in users_data])

def create_sample_analyses():
    """Create sample analysis jobs with realistic data"""
    users = list(User.objects.all())
    
    sample_analyses = [
        {
            'content_type': 'text',
            'text_content': 'BREAKING: Local hospital reports 90% reduction in COVID cases after implementing new treatment protocol using hydroxychloroquine and zinc supplements.',
            'status': 'completed',
            'verdict': 'likely_misinformation',
            'confidence': 0.87,
            'summary': 'Analysis detected misleading medical claims. Hydroxychloroquine efficacy claims contradict established medical research.',
            'evidence_data': [
                {
                    'evidence_type': 'source',
                    'method': 'medical_fact_check',
                    'score': 0.92,
                    'explanation': 'Multiple peer-reviewed studies contradict hydroxychloroquine efficacy claims for COVID-19 treatment.'
                },
                {
                    'evidence_type': 'forensic',
                    'method': 'language_analysis',
                    'score': 0.78,
                    'explanation': 'Text contains emotional language patterns typical of health misinformation.'
                }
            ]
        },
        {
            'content_type': 'text',
            'text_content': 'New archaeological discovery in Egypt reveals 4,000-year-old tomb with well-preserved artifacts, providing insights into Middle Kingdom burial practices.',
            'status': 'completed',
            'verdict': 'likely_true',
            'confidence': 0.91,
            'summary': 'Content appears factual and consistent with archaeological reporting standards.',
            'evidence_data': [
                {
                    'evidence_type': 'source',
                    'method': 'academic_verification',
                    'score': 0.89,
                    'explanation': 'Content structure and terminology consistent with legitimate archaeological reporting.'
                },
                {
                    'evidence_type': 'forensic',
                    'method': 'fact_pattern_analysis',
                    'score': 0.85,
                    'explanation': 'Factual claims align with known archaeological practices and historical context.'
                }
            ]
        },
        {
            'content_type': 'text',
            'text_content': 'Climate scientists warn that global temperatures could rise by 15 degrees Celsius within the next decade due to accelerating ice sheet collapse.',
            'status': 'completed',
            'verdict': 'likely_misinformation',
            'confidence': 0.94,
            'summary': 'Extreme temperature predictions significantly exceed scientific consensus and established climate models.',
            'evidence_data': [
                {
                    'evidence_type': 'source',
                    'method': 'climate_science_check',
                    'score': 0.96,
                    'explanation': 'Temperature rise claims far exceed IPCC projections and scientific consensus.'
                },
                {
                    'evidence_type': 'forensic',
                    'method': 'sensationalism_detection',
                    'score': 0.88,
                    'explanation': 'Language patterns indicate sensationalized climate reporting.'
                }
            ]
        },
        {
            'content_type': 'image',
            'status': 'completed',
            'verdict': 'likely_deepfake',
            'confidence': 0.82,
            'summary': 'Image forensics detected potential facial manipulation artifacts.',
            'evidence_data': [
                {
                    'evidence_type': 'forensic',
                    'method': 'deepfake_detection',
                    'score': 0.84,
                    'explanation': 'CNN model detected inconsistencies in facial features and lighting.'
                },
                {
                    'evidence_type': 'forensic',
                    'method': 'noise_analysis',
                    'score': 0.76,
                    'explanation': 'Noise patterns inconsistent across facial regions.'
                }
            ]
        },
        {
            'content_type': 'video',
            'status': 'completed',
            'verdict': 'inconclusive',
            'confidence': 0.65,
            'summary': 'Video analysis results inconclusive due to low resolution and compression artifacts.',
            'evidence_data': [
                {
                    'evidence_type': 'forensic',
                    'method': 'temporal_consistency',
                    'score': 0.68,
                    'explanation': 'Some temporal inconsistencies detected but within normal compression range.'
                },
                {
                    'evidence_type': 'forensic',
                    'method': 'audio_visual_sync',
                    'score': 0.72,
                    'explanation': 'Audio-visual synchronization appears normal.'
                }
            ]
        }
    ]
    
    created_jobs = []
    for i, analysis_data in enumerate(sample_analyses):
        user = users[i % len(users)]
        
        # Extract evidence data
        evidence_data = analysis_data.pop('evidence_data', [])
        
        # Create analysis job
        job = AnalysisJob.objects.create(
            user=user,
            **analysis_data,
            created_at=datetime.now() - timedelta(days=i),
            processing_time=2.5 + i * 0.5
        )
        
        # Create evidence records
        for evidence_item in evidence_data:
            Evidence.objects.create(
                job=job,
                **evidence_item
            )
        
        created_jobs.append(job)
    
    print(f"Created {len(created_jobs)} sample analysis jobs")
    return created_jobs

def create_sample_feedback():
    """Create sample user feedback"""
    jobs = AnalysisJob.objects.all()
    users = User.objects.all()
    
    feedback_data = [
        {
            'feedback_type': 'agree',
            'comment': 'Analysis correctly identified this as misinformation. Good work!'
        },
        {
            'feedback_type': 'disagree',
            'comment': 'I think this analysis might be too harsh. The source seemed credible to me.'
        },
        {
            'feedback_type': 'false_positive',
            'comment': 'This appears to be legitimate news that was incorrectly flagged.'
        }
    ]
    
    created_feedback = []
    for i, job in enumerate(jobs[:3]):  # Only create feedback for first 3 jobs
        user = users[i % len(users)]
        feedback = UserFeedback.objects.create(
            job=job,
            user=user,
            **feedback_data[i]
        )
        created_feedback.append(feedback)
    
    print(f"Created {len(created_feedback)} feedback records")
    return created_feedback

def main():
    """Main function to load all sample data"""
    print("Loading sample data for FactCheck AI...")
    
    # Create test users
    users = create_test_users()
    
    # Create sample analyses
    jobs = create_sample_analyses()
    
    # Create sample feedback
    feedback = create_sample_feedback()
    
    print("\nSample data loaded successfully!")
    print(f"- {User.objects.count()} users")
    print(f"- {AnalysisJob.objects.count()} analysis jobs")
    print(f"- {Evidence.objects.count()} evidence records")
    print(f"- {UserFeedback.objects.count()} feedback records")
    
    print("\nTest user credentials:")
    print("- journalist1 / testpass123")
    print("- researcher1 / testpass123")
    print("- factchecker1 / testpass123")

if __name__ == '__main__':
    main()
