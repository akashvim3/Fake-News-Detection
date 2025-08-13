"""
Real image forensics implementation using OpenCV and deep learning
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ExifTags
import logging
from typing import Dict, List, Any, Tuple
import os

logger = logging.getLogger(__name__)

class DeepfakeDetector(nn.Module):
    """CNN model for deepfake detection"""
    
    def __init__(self, num_classes: int = 2):
        super(DeepfakeDetector, self).__init__()
        
        # Use pre-trained EfficientNet as backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class ImageForensicsAnalyzer:
    """Comprehensive image forensics analyzer"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.deepfake_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.load_model()
    
    def load_model(self):
        """Load deepfake detection model"""
        try:
            self.deepfake_model = DeepfakeDetector()
            
            if self.model_path and os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.deepfake_model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from {self.model_path}")
            else:
                logger.warning("No pre-trained model found, using random weights")
            
            self.deepfake_model.to(self.device)
            self.deepfake_model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load deepfake model: {e}")
            self.deepfake_model = None
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Comprehensive image analysis"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            results = {
                'verdict': 'inconclusive',
                'confidence': 0.0,
                'evidence': [],
                'technical_details': {}
            }
            
            # 1. EXIF Analysis
            exif_results = self.analyze_exif(image_path)
            results['evidence'].extend(exif_results['evidence'])
            results['technical_details']['exif'] = exif_results['details']
            
            # 2. Error Level Analysis (ELA)
            ela_results = self.error_level_analysis(image)
            results['evidence'].append(ela_results)
            
            # 3. Noise Analysis
            noise_results = self.noise_analysis(image)
            results['evidence'].append(noise_results)
            
            # 4. Copy-Move Detection
            copy_move_results = self.copy_move_detection(image)
            results['evidence'].append(copy_move_results)
            
            # 5. Deepfake Detection (if model available)
            if self.deepfake_model:
                deepfake_results = self.deepfake_detection(image_path)
                results['evidence'].append(deepfake_results)
            
            # 6. Face Analysis
            face_results = self.face_analysis(image)
            if face_results:
                results['evidence'].append(face_results)
            
            # Combine results
            results = self.combine_forensic_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                'verdict': 'inconclusive',
                'confidence': 0.0,
                'evidence': [],
                'error': str(e)
            }
    
    def analyze_exif(self, image_path: str) -> Dict[str, Any]:
        """Analyze EXIF metadata"""
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()
            
            evidence = []
            details = {}
            
            if exif_data:
                exif = {ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS}
                details = exif
                
                # Check for suspicious patterns
                if 'Software' in exif:
                    software = exif['Software'].lower()
                    if any(editor in software for editor in ['photoshop', 'gimp', 'paint']):
                        evidence.append({
                            'type': 'forensic',
                            'method': 'exif_software',
                            'score': 0.7,
                            'explanation': f'Image editing software detected: {exif["Software"]}'
                        })
                
                if 'DateTime' not in exif:
                    evidence.append({
                        'type': 'forensic',
                        'method': 'exif_missing_datetime',
                        'score': 0.6,
                        'explanation': 'Missing timestamp in EXIF data'
                    })
            else:
                evidence.append({
                    'type': 'forensic',
                    'method': 'exif_missing',
                    'score': 0.8,
                    'explanation': 'No EXIF data found - possible manipulation'
                })
            
            return {'evidence': evidence, 'details': details}
            
        except Exception as e:
            return {'evidence': [], 'details': {'error': str(e)}}
    
    def error_level_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform Error Level Analysis (ELA)"""
        try:
            # Convert to PIL for JPEG compression
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Save with different quality levels
            import io
            buffer1 = io.BytesIO()
            buffer2 = io.BytesIO()
            
            pil_image.save(buffer1, format='JPEG', quality=90)
            pil_image.save(buffer2, format='JPEG', quality=95)
            
            # Load compressed images
            buffer1.seek(0)
            buffer2.seek(0)
            img1 = np.array(Image.open(buffer1))
            img2 = np.array(Image.open(buffer2))
            
            # Calculate difference
            diff = np.abs(img1.astype(float) - img2.astype(float))
            ela_score = np.mean(diff) / 255.0
            
            return {
                'type': 'forensic',
                'method': 'error_level_analysis',
                'score': min(ela_score * 10, 1.0),  # Normalize
                'explanation': f'ELA inconsistency score: {ela_score:.3f}'
            }
            
        except Exception as e:
            return {
                'type': 'forensic',
                'method': 'error_level_analysis',
                'score': 0.0,
                'explanation': f'ELA failed: {str(e)}'
            }
    
    def noise_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze noise patterns for inconsistencies"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur and subtract to get noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            
            # Divide image into blocks and analyze noise variance
            h, w = noise.shape
            block_size = 32
            variances = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = noise[i:i+block_size, j:j+block_size]
                    variances.append(np.var(block))
            
            # Calculate consistency score
            variance_std = np.std(variances)
            consistency_score = 1.0 - min(variance_std / 100.0, 1.0)
            
            return {
                'type': 'forensic',
                'method': 'noise_consistency',
                'score': consistency_score,
                'explanation': f'Noise pattern consistency: {consistency_score:.3f}'
            }
            
        except Exception as e:
            return {
                'type': 'forensic',
                'method': 'noise_consistency',
                'score': 0.0,
                'explanation': f'Noise analysis failed: {str(e)}'
            }
    
    def copy_move_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect copy-move forgery using SIFT features"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # SIFT detector
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            if descriptors is None or len(descriptors) < 10:
                return {
                    'type': 'forensic',
                    'method': 'copy_move_detection',
                    'score': 0.0,
                    'explanation': 'Insufficient features for copy-move detection'
                }
            
            # Find matches using FLANN
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(descriptors, descriptors, k=3)
            
            # Filter matches (exclude self-matches)
            good_matches = []
            for match_group in matches:
                if len(match_group) >= 2:
                    m, n = match_group[0], match_group[1]
                    if m.distance < 0.7 * n.distance and m.queryIdx != m.trainIdx:
                        good_matches.append(m)
            
            # Calculate suspicion score based on number of similar regions
            suspicion_score = min(len(good_matches) / 50.0, 1.0)
            
            return {
                'type': 'forensic',
                'method': 'copy_move_detection',
                'score': suspicion_score,
                'explanation': f'Found {len(good_matches)} potential copy-move regions'
            }
            
        except Exception as e:
            return {
                'type': 'forensic',
                'method': 'copy_move_detection',
                'score': 0.0,
                'explanation': f'Copy-move detection failed: {str(e)}'
            }
    
    def deepfake_detection(self, image_path: str) -> Dict[str, Any]:
        """Detect deepfakes using CNN model"""
        try:
            if not self.deepfake_model:
                return {
                    'type': 'forensic',
                    'method': 'deepfake_detection',
                    'score': 0.0,
                    'explanation': 'Deepfake model not available'
                }
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.deepfake_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                deepfake_prob = probabilities[0][1].item()  # Assuming class 1 is deepfake
            
            return {
                'type': 'forensic',
                'method': 'deepfake_detection',
                'score': deepfake_prob,
                'explanation': f'CNN deepfake probability: {deepfake_prob:.3f}'
            }
            
        except Exception as e:
            return {
                'type': 'forensic',
                'method': 'deepfake_detection',
                'score': 0.0,
                'explanation': f'Deepfake detection failed: {str(e)}'
            }
    
    def face_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze faces for manipulation signs"""
        try:
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Analyze each face
            face_scores = []
            for (x, y, w, h) in faces:
                face_roi = image[y:y+h, x:x+w]
                
                # Simple face quality analysis
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
                
                # Normalize sharpness score
                sharpness_score = min(laplacian_var / 1000.0, 1.0)
                face_scores.append(sharpness_score)
            
            avg_face_quality = np.mean(face_scores)
            
            return {
                'type': 'forensic',
                'method': 'face_analysis',
                'score': avg_face_quality,
                'explanation': f'Detected {len(faces)} faces, avg quality: {avg_face_quality:.3f}'
            }
            
        except Exception as e:
            return {
                'type': 'forensic',
                'method': 'face_analysis',
                'score': 0.0,
                'explanation': f'Face analysis failed: {str(e)}'
            }
    
    def combine_forensic_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all forensic evidence into final verdict"""
        evidence = results['evidence']
        
        if not evidence:
            return results
        
        # Calculate weighted average of scores
        total_score = 0
        total_weight = 0
        
        weights = {
            'deepfake_detection': 0.4,
            'exif_missing': 0.2,
            'exif_software': 0.15,
            'noise_consistency': 0.1,
            'copy_move_detection': 0.1,
            'error_level_analysis': 0.05
        }
        
        for item in evidence:
            method = item.get('method', '')
            score = item.get('score', 0)
            weight = weights.get(method, 0.05)
            
            total_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            final_confidence = total_score / total_weight
        else:
            final_confidence = 0.0
        
        # Determine verdict
        if final_confidence > 0.7:
            verdict = 'likely_deepfake'
        elif final_confidence < 0.3:
            verdict = 'likely_true'
        else:
            verdict = 'inconclusive'
        
        results['verdict'] = verdict
        results['confidence'] = final_confidence
        
        return results

if __name__ == "__main__":
    # Example usage
    analyzer = ImageForensicsAnalyzer()
    result = analyzer.analyze_image("test_image.jpg")
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.2f}")
