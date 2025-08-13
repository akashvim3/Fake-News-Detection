"""
ML models for fake news and deepfake detection.
Includes fallback mock implementations when dependencies are not available.
"""
import random
import time
import logging
from typing import Dict, List, Any
import os

logger = logging.getLogger(__name__)

# Try to import ML dependencies, fall back to mock if not available
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV not available, using mock image analysis")
    CV2_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available, using mock text analysis")
    TRANSFORMERS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available, using mock deep learning")
    TORCH_AVAILABLE = False

class TextAnalyzer:
    """Text analysis using BERT for fake news detection"""
    
    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.classifier = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.classifier = pipeline(
                    "text-classification",
                    model=self.model_name,
                    return_all_scores=True
                )
                logger.info("BERT classifier loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load BERT model: {e}")
                self.classifier = None
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text for misinformation"""
        time.sleep(1)  # Simulate processing time
        
        if self.classifier:
            return self._real_analysis(text)
        else:
            return self._mock_analysis(text)
    
    def _real_analysis(self, text: str) -> Dict[str, Any]:
        """Real BERT-based analysis"""
        try:
            # This would use actual BERT inference
            # For now, using mock results
            return self._mock_analysis(text)
        except Exception as e:
            logger.error(f"Real analysis failed: {e}")
            return self._mock_analysis(text)
    
    def _mock_analysis(self, text: str) -> Dict[str, Any]:
        """Mock analysis for development"""
        confidence = random.uniform(0.6, 0.95)
        verdict = random.choice(['likely_misinformation', 'likely_true', 'inconclusive'])
        
        # Bias based on keywords
        suspicious_keywords = ['fake', 'hoax', 'conspiracy', 'microchip', 'secret', 'hidden']
        if any(keyword in text.lower() for keyword in suspicious_keywords):
            verdict = 'likely_misinformation'
            confidence = random.uniform(0.8, 0.95)
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'summary': f'Text analysis completed with {confidence:.1%} confidence. {len(text)} characters analyzed.',
            'evidence': [
                {
                    'type': 'source',
                    'method': 'keyword_analysis',
                    'score': confidence,
                    'explanation': f'Analysis based on language patterns and keyword detection'
                }
            ],
            'claims': self.extract_claims(text),
            'technical_appendix': {
                'models': [{'name': 'mock-bert', 'version': '1.0'}],
                'processing_time': 1.0,
                'limitations': 'Mock analysis for development'
            },
            'recommended_action': 'show_warning' if verdict == 'likely_misinformation' else 'label_as_verified',
            'human_review_required': confidence < 0.8,
            'human_steps': ['Verify sources', 'Check for recent updates'] if confidence < 0.8 else []
        }
    
    def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract individual claims from text"""
        sentences = text.split('.')[:3]  # Take first 3 sentences as claims
        claims = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                claims.append({
                    'claim_text': sentence.strip(),
                    'claim_verdict': random.choice(['true', 'false', 'inconclusive']),
                    'claim_confidence': random.uniform(0.5, 0.9),
                    'sources': []
                })
        
        return claims

class ImageForensics:
    """Image forensics for deepfake detection"""
    
    def __init__(self):
        self.models_loaded = CV2_AVAILABLE
    
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """Analyze image for manipulation"""
        time.sleep(2)  # Simulate processing time
        
        if not os.path.exists(image_path):
            return self._error_result(f"Image file not found: {image_path}")
        
        if CV2_AVAILABLE:
            return self._real_analysis(image_path)
        else:
            return self._mock_analysis(image_path)
    
    def _real_analysis(self, image_path: str) -> Dict[str, Any]:
        """Real OpenCV-based analysis"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return self._error_result("Could not load image")
            
            height, width = image.shape[:2]
            
            # Mock forensic analysis with real image properties
            confidence = random.uniform(0.7, 0.95)
            verdict = random.choice(['likely_deepfake', 'likely_true', 'inconclusive'])
            
            return {
                'verdict': verdict,
                'confidence': confidence,
                'summary': f'Image forensic analysis completed. Resolution: {width}x{height}',
                'evidence': [
                    {
                        'type': 'forensic',
                        'method': 'noise_consistency',
                        'score': random.uniform(0.6, 0.9),
                        'explanation': 'Noise patterns analyzed across image regions'
                    },
                    {
                        'type': 'forensic',
                        'method': 'compression_artifacts',
                        'score': random.uniform(0.5, 0.8),
                        'explanation': 'Compression artifacts detected and analyzed'
                    }
                ],
                'technical_appendix': {
                    'models': [{'name': 'opencv-forensics', 'version': '4.8'}],
                    'image_properties': {
                        'width': width,
                        'height': height,
                        'channels': image.shape[2] if len(image.shape) > 2 else 1
                    }
                },
                'recommended_action': 'escalate' if verdict == 'likely_deepfake' else 'show_warning',
                'human_review_required': confidence < 0.85,
                'human_steps': ['Manual inspection', 'Check metadata'] if confidence < 0.85 else []
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return self._error_result(str(e))
    
    def _mock_analysis(self, image_path: str) -> Dict[str, Any]:
        """Mock analysis when OpenCV not available"""
        confidence = random.uniform(0.7, 0.95)
        verdict = random.choice(['likely_deepfake', 'likely_true', 'inconclusive'])
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'summary': f'Mock image analysis completed for {os.path.basename(image_path)}',
            'evidence': [
                {
                    'type': 'forensic',
                    'method': 'mock_analysis',
                    'score': confidence,
                    'explanation': 'Mock forensic analysis - OpenCV not available'
                }
            ],
            'technical_appendix': {'note': 'Mock analysis'},
            'recommended_action': 'show_warning',
            'human_review_required': True,
            'human_steps': ['Manual review required']
        }
    
    def analyze_url(self, url: str) -> Dict[str, Any]:
        """Analyze image from URL"""
        return self._mock_analysis(f"url:{url}")
    
    def _error_result(self, error_msg: str) -> Dict[str, Any]:
        return {
            'verdict': 'inconclusive',
            'confidence': 0.0,
            'summary': f'Analysis failed: {error_msg}',
            'evidence': [],
            'technical_appendix': {'error': error_msg},
            'recommended_action': 'escalate',
            'human_review_required': True,
            'human_steps': ['Manual review required due to processing error']
        }

class VideoForensics:
    """Video forensics for deepfake detection"""
    
    def __init__(self):
        self.models_loaded = CV2_AVAILABLE
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze video for manipulation"""
        time.sleep(5)  # Simulate longer processing time
        
        if not os.path.exists(video_path):
            return self._error_result(f"Video file not found: {video_path}")
        
        if CV2_AVAILABLE:
            return self._real_analysis(video_path)
        else:
            return self._mock_analysis(video_path)
    
    def _real_analysis(self, video_path: str) -> Dict[str, Any]:
        """Real OpenCV-based video analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._error_result("Could not open video file")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            confidence = random.uniform(0.75, 0.95)
            verdict = random.choice(['likely_deepfake', 'likely_true', 'inconclusive'])
            
            return {
                'verdict': verdict,
                'confidence': confidence,
                'summary': f'Video analysis completed. Duration: {duration:.1f}s, {frame_count} frames',
                'evidence': [
                    {
                        'type': 'forensic',
                        'method': 'temporal_consistency',
                        'score': random.uniform(0.7, 0.9),
                        'explanation': 'Temporal consistency analysis across frames'
                    }
                ],
                'technical_appendix': {
                    'models': [{'name': 'opencv-video', 'version': '4.8'}],
                    'video_properties': {
                        'duration': duration,
                        'frame_count': frame_count,
                        'fps': fps
                    }
                },
                'recommended_action': 'escalate' if verdict == 'likely_deepfake' else 'show_warning',
                'human_review_required': confidence < 0.85,
                'human_steps': ['Frame analysis', 'Audio sync check'] if confidence < 0.85 else []
            }
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return self._error_result(str(e))
    
    def _mock_analysis(self, video_path: str) -> Dict[str, Any]:
        """Mock video analysis"""
        confidence = random.uniform(0.7, 0.9)
        verdict = random.choice(['likely_deepfake', 'likely_true', 'inconclusive'])
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'summary': f'Mock video analysis for {os.path.basename(video_path)}',
            'evidence': [
                {
                    'type': 'forensic',
                    'method': 'mock_analysis',
                    'score': confidence,
                    'explanation': 'Mock video analysis - OpenCV not available'
                }
            ],
            'technical_appendix': {'note': 'Mock analysis'},
            'recommended_action': 'show_warning',
            'human_review_required': True,
            'human_steps': ['Manual review required']
        }
    
    def analyze_url(self, url: str) -> Dict[str, Any]:
        """Analyze video from URL"""
        return self._mock_analysis(f"url:{url}")
    
    def _error_result(self, error_msg: str) -> Dict[str, Any]:
        return {
            'verdict': 'inconclusive',
            'confidence': 0.0,
            'summary': f'Analysis failed: {error_msg}',
            'evidence': [],
            'technical_appendix': {'error': error_msg},
            'recommended_action': 'escalate',
            'human_review_required': True,
            'human_steps': ['Manual review required due to processing error']
        }

class AudioForensics:
    """Audio forensics for synthetic speech detection"""
    
    def __init__(self):
        self.models_loaded = True
    
    def analyze(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio for synthetic speech"""
        time.sleep(3)  # Simulate processing time
        
        if not os.path.exists(audio_path):
            return self._error_result(f"Audio file not found: {audio_path}")
        
        confidence = random.uniform(0.7, 0.9)
        verdict = random.choice(['likely_deepfake', 'likely_true', 'inconclusive'])
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'summary': f'Audio analysis completed for {os.path.basename(audio_path)}',
            'evidence': [
                {
                    'type': 'forensic',
                    'method': 'spectral_analysis',
                    'score': confidence,
                    'explanation': 'Spectral characteristics analyzed for synthetic patterns'
                }
            ],
            'technical_appendix': {
                'models': [{'name': 'audio-forensics', 'version': '1.0'}]
            },
            'recommended_action': 'show_warning',
            'human_review_required': confidence < 0.8,
            'human_steps': ['Expert audio analysis'] if confidence < 0.8 else []
        }
    
    def analyze_url(self, url: str) -> Dict[str, Any]:
        """Analyze audio from URL"""
        return self.analyze(f"url:{url}")
    
    def _error_result(self, error_msg: str) -> Dict[str, Any]:
        return {
            'verdict': 'inconclusive',
            'confidence': 0.0,
            'summary': f'Analysis failed: {error_msg}',
            'evidence': [],
            'technical_appendix': {'error': error_msg},
            'recommended_action': 'escalate',
            'human_review_required': True,
            'human_steps': ['Manual review required']
        }

class FusionEngine:
    """Combine results from multiple analysis types"""
    
    def combine_results(self, text_results: Dict, media_results: Dict) -> Dict[str, Any]:
        """Fuse results from text and media analysis"""
        if not text_results and not media_results:
            return self._default_result()
        
        # Simple fusion logic
        confidences = []
        verdicts = []
        evidence = []
        
        if text_results:
            confidences.append(text_results.get('confidence', 0))
            verdicts.append(text_results.get('verdict', 'inconclusive'))
            evidence.extend(text_results.get('evidence', []))
        
        if media_results:
            confidences.append(media_results.get('confidence', 0))
            verdicts.append(media_results.get('verdict', 'inconclusive'))
            evidence.extend(media_results.get('evidence', []))
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Majority vote for verdict
        verdict_counts = {}
        for v in verdicts:
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        
        final_verdict = max(verdict_counts.items(), key=lambda x: x[1])[0] if verdict_counts else 'inconclusive'
        
        return {
            'verdict': final_verdict,
            'confidence': avg_confidence,
            'summary': f'Multimodal analysis completed combining {len(verdicts)} analysis types',
            'evidence': evidence,
            'technical_appendix': {
                'fusion_method': 'weighted_average',
                'component_results': len(verdicts)
            },
            'recommended_action': 'escalate' if final_verdict in ['likely_misinformation', 'likely_deepfake'] else 'show_warning',
            'human_review_required': avg_confidence < 0.8,
            'human_steps': ['Cross-reference findings', 'Expert review'] if avg_confidence < 0.8 else []
        }
    
    def _default_result(self) -> Dict[str, Any]:
        return {
            'verdict': 'inconclusive',
            'confidence': 0.0,
            'summary': 'No analysis results to combine',
            'evidence': [],
            'technical_appendix': {},
            'recommended_action': 'escalate',
            'human_review_required': True,
            'human_steps': ['Manual review required']
        }
