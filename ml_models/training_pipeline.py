"""
Training pipeline for fake news detection models
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
import logging
from typing import List, Tuple, Dict
import json

from .text_classifier import FakeNewsClassifier, FakeNewsTrainer
from .image_forensics import DeepfakeDetector

logger = logging.getLogger(__name__)

class FakeNewsDataset:
    """Dataset loader for fake news detection"""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        
    def load_liar_dataset(self) -> Tuple[List[str], List[int]]:
        """Load LIAR dataset"""
        try:
            # Download and load LIAR dataset
            # This would typically load from TSV files
            train_file = os.path.join(self.data_dir, "liar_train.tsv")
            
            if not os.path.exists(train_file):
                logger.warning("LIAR dataset not found, using sample data")
                return self._get_sample_data()
            
            df = pd.read_csv(train_file, sep='\t', header=None)
            texts = df[2].tolist()  # Statement column
            labels = df[1].apply(lambda x: 1 if x in ['false', 'pants-fire'] else 0).tolist()
            
            return texts, labels
            
        except Exception as e:
            logger.error(f"Failed to load LIAR dataset: {e}")
            return self._get_sample_data()
    
    def load_fever_dataset(self) -> Tuple[List[str], List[int]]:
        """Load FEVER dataset"""
        try:
            fever_file = os.path.join(self.data_dir, "fever_train.jsonl")
            
            if not os.path.exists(fever_file):
                logger.warning("FEVER dataset not found, using sample data")
                return self._get_sample_data()
            
            texts = []
            labels = []
            
            with open(fever_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    texts.append(data['claim'])
                    # Convert FEVER labels to binary
                    label = 1 if data['label'] == 'REFUTES' else 0
                    labels.append(label)
            
            return texts, labels
            
        except Exception as e:
            logger.error(f"Failed to load FEVER dataset: {e}")
            return self._get_sample_data()
    
    def _get_sample_data(self) -> Tuple[List[str], List[int]]:
        """Generate sample training data"""
        fake_samples = [
            "COVID-19 vaccines contain microchips for government tracking",
            "5G towers are spreading coronavirus and causing cancer",
            "The earth is flat and NASA is hiding the truth from us",
            "Drinking bleach can cure COVID-19 and other diseases",
            "Climate change is a hoax created by scientists for money",
            "Vaccines cause autism in children according to studies",
            "The moon landing was filmed in a Hollywood studio",
            "Chemtrails are being used for population control",
            "Fluoride in water is a government mind control chemical",
            "GMO foods are designed to make people sick and dependent"
        ]
        
        real_samples = [
            "Scientists develop new method for early cancer detection using AI",
            "Renewable energy sources now account for 30% of global electricity",
            "New archaeological discovery reveals ancient trade routes",
            "Researchers find potential treatment for Alzheimer's disease",
            "Climate data shows continued warming trend over past decade",
            "Vaccination rates correlate with reduced disease transmission",
            "Space telescope discovers potentially habitable exoplanet",
            "Study shows benefits of regular exercise for mental health",
            "New water purification technology could help developing nations",
            "Genetic research advances understanding of rare diseases"
        ]
        
        texts = fake_samples + real_samples
        labels = [1] * len(fake_samples) + [0] * len(real_samples)
        
        # Expand dataset with variations
        expanded_texts = []
        expanded_labels = []
        
        for text, label in zip(texts, labels):
            expanded_texts.append(text)
            expanded_labels.append(label)
            
            # Add variations
            variations = [
                f"Breaking news: {text}",
                f"According to sources, {text.lower()}",
                f"Reports suggest that {text.lower()}",
                f"New study reveals: {text.lower()}"
            ]
            
            for variation in variations:
                expanded_texts.append(variation)
                expanded_labels.append(label)
        
        return expanded_texts, expanded_labels

class ImageDataset(Dataset):
    """Dataset for deepfake detection training"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

class TrainingPipeline:
    """Complete training pipeline for all models"""
    
    def __init__(self, output_dir: str = "models/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.dataset_loader = FakeNewsDataset()
    
    def train_text_classifier(self, epochs: int = 3, batch_size: int = 16):
        """Train BERT-based fake news classifier"""
        logger.info("Starting text classifier training...")
        
        # Load data
        texts, labels = self.dataset_loader.load_liar_dataset()
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
        
        # Initialize trainer
        trainer = FakeNewsTrainer()
        
        # Train model
        model_output_dir = os.path.join(self.output_dir, "fake_news_bert")
        trained_model = trainer.train(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            output_dir=model_output_dir,
            num_epochs=epochs,
            batch_size=batch_size
        )
        
        # Evaluate model
        self._evaluate_text_model(trained_model, val_texts, val_labels)
        
        logger.info(f"Text classifier saved to {model_output_dir}")
        
        return model_output_dir
    
    def train_deepfake_detector(self, data_dir: str, epochs: int = 10, batch_size: int = 32):
        """Train deepfake detection model"""
        logger.info("Starting deepfake detector training...")
        
        # This would load actual deepfake datasets like FaceForensics++
        # For demo, we'll create a mock training loop
        
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        model = DeepfakeDetector()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Mock training (replace with actual dataset loading)
        logger.info("Mock training for deepfake detector...")
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            # Mock batch training
            for batch_idx in range(10):  # Mock 10 batches
                # Generate random data (replace with real data loader)
                inputs = torch.randn(batch_size, 3, 224, 224).to(device)
                labels = torch.randint(0, 2, (batch_size,)).to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / 10
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Save model
        model_path = os.path.join(self.output_dir, "deepfake_detector.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epochs,
        }, model_path)
        
        logger.info(f"Deepfake detector saved to {model_path}")
        
        return model_path
    
    def _evaluate_text_model(self, trainer, val_texts: List[str], val_labels: List[int]):
        """Evaluate text classification model"""
        try:
            # Get predictions
            predictions = trainer.predict(trainer.eval_dataset)
            pred_labels = np.argmax(predictions.predictions, axis=1)
            
            # Print evaluation metrics
            report = classification_report(val_labels, pred_labels, 
                                         target_names=['Real', 'Fake'])
            logger.info(f"Classification Report:\n{report}")
            
            # Confusion matrix
            cm = confusion_matrix(val_labels, pred_labels)
            logger.info(f"Confusion Matrix:\n{cm}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
    
    def create_model_config(self):
        """Create configuration file for models"""
        config = {
            "text_classifier": {
                "model_type": "bert",
                "model_name": "bert-base-uncased",
                "num_labels": 2,
                "max_length": 512,
                "threshold": 0.7
            },
            "deepfake_detector": {
                "model_type": "efficientnet",
                "input_size": [224, 224],
                "num_classes": 2,
                "threshold": 0.8
            },
            "ensemble": {
                "weights": {
                    "text": 0.4,
                    "image": 0.6
                },
                "fusion_method": "weighted_average"
            }
        }
        
        config_path = os.path.join(self.output_dir, "model_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model configuration saved to {config_path}")
        
        return config_path

def main():
    """Main training script"""
    pipeline = TrainingPipeline()
    
    # Train text classifier
    text_model_path = pipeline.train_text_classifier(epochs=3, batch_size=16)
    
    # Train deepfake detector
    deepfake_model_path = pipeline.train_deepfake_detector("data/deepfakes/", epochs=5)
    
    # Create configuration
    config_path = pipeline.create_model_config()
    
    print(f"Training completed!")
    print(f"Text model: {text_model_path}")
    print(f"Deepfake model: {deepfake_model_path}")
    print(f"Config: {config_path}")

if __name__ == "__main__":
    main()
