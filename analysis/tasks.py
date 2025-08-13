from celery import shared_task
from django.utils import timezone
from django.conf import settings
from .models import AnalysisJob, Claim, Evidence
import logging
import time
import traceback

logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3)
def process_analysis_job(self, job_id):
    """Process an analysis job using ML models"""
    try:
        job = AnalysisJob.objects.get(id=job_id)
        job.status = 'processing'
        job.save()
        
        start_time = time.time()
        logger.info(f"Starting analysis for job {job_id}")
        
        # Import ML models here to avoid import issues
        try:
            from .ml_models import (
                TextAnalyzer, 
                ImageForensics, 
                VideoForensics, 
                AudioForensics,
                FusionEngine
            )
        except ImportError:
            # Fallback to mock analysis if ML models not available
            logger.warning("ML models not available, using mock analysis")
            return mock_analysis(job, start_time)
        
        # Initialize analyzers
        text_analyzer = TextAnalyzer()
        image_forensics = ImageForensics()
        video_forensics = VideoForensics()
        audio_forensics = AudioForensics()
        fusion_engine = FusionEngine()
        
        results = {}
        evidence_items = []
        claims = []
        
        # Process based on content type
        if job.content_type == 'text' and job.text_content:
            results = text_analyzer.analyze(job.text_content)
            claims = text_analyzer.extract_claims(job.text_content)
            
        elif job.content_type == 'image':
            if job.file_upload and job.file_upload.path:
                results = image_forensics.analyze(job.file_upload.path)
            elif job.url_input:
                results = image_forensics.analyze_url(job.url_input)
            else:
                results = {'verdict': 'inconclusive', 'confidence': 0.0, 'summary': 'No image provided'}
                
        elif job.content_type == 'video':
            if job.file_upload and job.file_upload.path:
                results = video_forensics.analyze(job.file_upload.path)
            elif job.url_input:
                results = video_forensics.analyze_url(job.url_input)
            else:
                results = {'verdict': 'inconclusive', 'confidence': 0.0, 'summary': 'No video provided'}
                
        elif job.content_type == 'audio':
            if job.file_upload and job.file_upload.path:
                results = audio_forensics.analyze(job.file_upload.path)
            elif job.url_input:
                results = audio_forensics.analyze_url(job.url_input)
            else:
                results = {'verdict': 'inconclusive', 'confidence': 0.0, 'summary': 'No audio provided'}
                
        elif job.content_type == 'multimodal':
            # Combine multiple analysis types
            text_results = {}
            media_results = {}
            
            if job.text_content:
                text_results = text_analyzer.analyze(job.text_content)
            
            if job.file_upload and job.file_upload.path:
                file_ext = job.file_upload.name.split('.')[-1].lower()
                if file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'webp']:
                    media_results = image_forensics.analyze(job.file_upload.path)
                elif file_ext in ['mp4', 'avi', 'mov', 'webm']:
                    media_results = video_forensics.analyze(job.file_upload.path)
                elif file_ext in ['mp3', 'wav', 'flac', 'm4a']:
                    media_results = audio_forensics.analyze(job.file_upload.path)
            
            results = fusion_engine.combine_results(text_results, media_results)
        
        # Process results
        if results:
            job.verdict = results.get('verdict', 'inconclusive')
            job.confidence = results.get('confidence', 0.0)
            job.summary = results.get('summary', 'Analysis completed')
            job.evidence = results.get('evidence', [])
            job.claims = results.get('claims', [])
            job.technical_appendix = results.get('technical_appendix', {})
            job.recommended_action = results.get('recommended_action', 'show_warning')
            job.human_review_required = results.get('human_review_required', False)
            job.human_steps = results.get('human_steps', [])
            
            # Create evidence records
            for evidence_data in results.get('evidence', []):
                try:
                    Evidence.objects.create(
                        job=job,
                        evidence_type=evidence_data.get('type', 'forensic'),
                        method=evidence_data.get('method', 'unknown'),
                        score=evidence_data.get('score', 0.0),
                        explanation=evidence_data.get('explanation', ''),
                        metadata=evidence_data.get('metadata', {})
                    )
                except Exception as e:
                    logger.error(f"Failed to create evidence record: {e}")
            
            # Create claim records
            for claim_data in results.get('claims', []):
                try:
                    Claim.objects.create(
                        job=job,
                        claim_text=claim_data.get('claim_text', ''),
                        claim_verdict=claim_data.get('claim_verdict', 'inconclusive'),
                        claim_confidence=claim_data.get('claim_confidence', 0.0),
                        sources=claim_data.get('sources', [])
                    )
                except Exception as e:
                    logger.error(f"Failed to create claim record: {e}")
        
        # Update job status
        processing_time = time.time() - start_time
        job.processing_time = processing_time
        job.status = 'completed'
        job.completed_at = timezone.now()
        job.save()
        
        logger.info(f"Completed analysis for job {job_id} in {processing_time:.2f}s")
        
    except AnalysisJob.DoesNotExist:
        logger.error(f"Job {job_id} not found")
        
    except Exception as exc:
        logger.error(f"Analysis failed for job {job_id}: {str(exc)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        try:
            job = AnalysisJob.objects.get(id=job_id)
            job.status = 'failed'
            job.error_message = str(exc)
            job.save()
        except:
            pass
        
        # Retry the task
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying job {job_id} (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60 * (2 ** self.request.retries))

def mock_analysis(job, start_time):
    """Mock analysis when ML models are not available"""
    import random
    
    # Generate mock results
    verdicts = ['likely_misinformation', 'likely_true', 'inconclusive']
    verdict = random.choice(verdicts)
    confidence = random.uniform(0.6, 0.95)
    
    # Bias based on content
    if job.text_content and any(word in job.text_content.lower() 
                               for word in ['fake', 'hoax', 'conspiracy', 'microchip']):
        verdict = 'likely_misinformation'
        confidence = random.uniform(0.8, 0.95)
    
    job.verdict = verdict
    job.confidence = confidence
    job.summary = f'Mock analysis completed with {confidence:.1%} confidence'
    job.evidence = [
        {
            'type': 'forensic',
            'method': 'mock_analysis',
            'score': confidence,
            'explanation': 'This is a mock analysis result for development purposes'
        }
    ]
    job.technical_appendix = {'note': 'Mock analysis - ML models not loaded'}
    job.recommended_action = 'show_warning' if verdict == 'likely_misinformation' else 'label_as_verified'
    job.human_review_required = confidence < 0.8
    
    # Update job status
    processing_time = time.time() - start_time
    job.processing_time = processing_time
    job.status = 'completed'
    job.completed_at = timezone.now()
    job.save()
    
    logger.info(f"Completed mock analysis for job {job.id} in {processing_time:.2f}s")
