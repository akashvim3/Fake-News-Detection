"""
Test cases for the FactCheck API
"""
import pytest
import json
from django.test import TestCase, Client
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from analysis.models import AnalysisJob
from django.core.files.uploadedfile import SimpleUploadedFile

class FactCheckAPITestCase(TestCase):
    """Test cases for FactCheck API endpoints"""
    
    def setUp(self):
        """Set up test data"""
        self.client = Client()
        
        # Create test user
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        # Create auth token
        self.token = Token.objects.create(user=self.user)
        
        # Set auth header
        self.auth_headers = {
            'HTTP_AUTHORIZATION': f'Token {self.token.key}'
        }
    
    def test_text_analysis_submission(self):
        """Test text analysis submission"""
        data = {
            'content_type': 'text',
            'text_content': 'Breaking: Scientists discover vaccines contain microchips for government surveillance.',
            'metadata': json.dumps({'source': 'social_media'})
        }
        
        response = self.client.post(
            '/api/v1/analysis/analyze/',
            data=data,
            **self.auth_headers
        )
        
        self.assertEqual(response.status_code, 202)
        response_data = response.json()
        self.assertIn('job_id', response_data)
        self.assertEqual(response_data['status'], 'pending')
        
        # Check job was created
        job = AnalysisJob.objects.get(id=response_data['job_id'])
        self.assertEqual(job.content_type, 'text')
        self.assertEqual(job.user, self.user)
    
    def test_image_analysis_submission(self):
        """Test image analysis submission"""
        # Create mock image file
        image_content = b'fake_image_content'
        image_file = SimpleUploadedFile(
            "test_image.jpg",
            image_content,
            content_type="image/jpeg"
        )
        
        data = {
            'content_type': 'image',
            'file_upload': image_file,
            'metadata': json.dumps({'source': 'upload'})
        }
        
        response = self.client.post(
            '/api/v1/analysis/analyze/',
            data=data,
            **self.auth_headers
        )
        
        self.assertEqual(response.status_code, 202)
        response_data = response.json()
        self.assertIn('job_id', response_data)
    
    def test_job_status_retrieval(self):
        """Test retrieving job status"""
        # Create test job
        job = AnalysisJob.objects.create(
            user=self.user,
            content_type='text',
            text_content='Test content',
            status='completed',
            verdict='likely_misinformation',
            confidence=0.85,
            summary='Test analysis completed'
        )
        
        response = self.client.get(
            f'/api/v1/analysis/{job.id}/status/',
            **self.auth_headers
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data['verdict'], 'likely_misinformation')
        self.assertEqual(response_data['confidence'], 0.85)
    
    def test_user_feedback_submission(self):
        """Test user feedback submission"""
        # Create test job
        job = AnalysisJob.objects.create(
            user=self.user,
            content_type='text',
            text_content='Test content',
            status='completed',
            verdict='likely_misinformation',
            confidence=0.85
        )
        
        feedback_data = {
            'feedback_type': 'disagree',
            'comment': 'This analysis seems incorrect'
        }
        
        response = self.client.post(
            f'/api/v1/analysis/{job.id}/feedback/',
            data=json.dumps(feedback_data),
            content_type='application/json',
            **self.auth_headers
        )
        
        self.assertEqual(response.status_code, 201)
        
        # Check feedback was created
        self.assertTrue(job.feedback.filter(user=self.user).exists())
    
    def test_analysis_stats(self):
        """Test analysis statistics endpoint"""
        # Create test jobs
        AnalysisJob.objects.create(
            user=self.user,
            content_type='text',
            status='completed'
        )
        AnalysisJob.objects.create(
            user=self.user,
            content_type='image',
            status='pending'
        )
        
        response = self.client.get(
            '/api/v1/analysis/stats/',
            **self.auth_headers
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data['total_analyses'], 2)
        self.assertEqual(response_data['completed'], 1)
        self.assertEqual(response_data['pending'], 1)
    
    def test_unauthorized_access(self):
        """Test unauthorized access is blocked"""
        data = {
            'content_type': 'text',
            'text_content': 'Test content'
        }
        
        response = self.client.post('/api/v1/analysis/analyze/', data=data)
        self.assertEqual(response.status_code, 401)
    
    def test_invalid_content_type(self):
        """Test invalid content type handling"""
        data = {
            'content_type': 'invalid_type',
            'text_content': 'Test content'
        }
        
        response = self.client.post(
            '/api/v1/analysis/analyze/',
            data=data,
            **self.auth_headers
        )
        
        self.assertEqual(response.status_code, 400)

class MLModelTestCase(TestCase):
    """Test cases for ML models"""
    
    def test_text_classifier_prediction(self):
        """Test text classifier prediction"""
        from ml_models.text_classifier import FakeNewsClassifier
        
        classifier = FakeNewsClassifier()
        classifier.load_model()  # Load mock model
        
        test_text = "Breaking: Scientists discover vaccines contain microchips"
        result = classifier.predict(test_text)
        
        self.assertIn('verdict', result)
        self.assertIn('confidence', result)
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_claim_extraction(self):
        """Test claim extraction from text"""
        from ml_models.text_classifier import FakeNewsClassifier
        
        classifier = FakeNewsClassifier()
        text = "The earth is flat. NASA is hiding the truth. Scientists are lying to us."
        
        claims = classifier.extract_claims(text)
        
        self.assertIsInstance(claims, list)
        self.assertGreater(len(claims), 0)
        self.assertIn("The earth is flat", claims[0])

if __name__ == '__main__':
    pytest.main([__file__])
