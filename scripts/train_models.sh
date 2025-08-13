#!/bin/bash

# Training script for all models

echo "Starting model training pipeline..."

# Create necessary directories
mkdir -p data/
mkdir -p models/
mkdir -p logs/

# Download datasets
echo "Downloading datasets..."
python scripts/download_datasets.py

# Train text classifier
echo "Training BERT-based fake news classifier..."
python -c "
from ml_models.training_pipeline import TrainingPipeline
pipeline = TrainingPipeline()
pipeline.train_text_classifier(epochs=3, batch_size=16)
"

# Train deepfake detector (mock training)
echo "Training deepfake detector..."
python -c "
from ml_models.training_pipeline import TrainingPipeline
pipeline = TrainingPipeline()
pipeline.train_deepfake_detector('data/deepfakes/', epochs=5)
"

# Create model configuration
echo "Creating model configuration..."
python -c "
from ml_models.training_pipeline import TrainingPipeline
pipeline = TrainingPipeline()
pipeline.create_model_config()
"

echo "Training completed!"
echo "Models saved in: models/"
echo "Logs saved in: logs/"

# Test models
echo "Testing trained models..."
python -c "
from ml_models.text_classifier import FakeNewsClassifier
from ml_models.image_forensics import ImageForensicsAnalyzer

# Test text classifier
classifier = FakeNewsClassifier()
classifier.load_model('models/fake_news_bert/')
result = classifier.predict('Breaking: Scientists discover vaccines contain microchips')
print(f'Text Analysis - Verdict: {result[\"verdict\"]}, Confidence: {result[\"confidence\"]:.2f}')

# Test image forensics
analyzer = ImageForensicsAnalyzer('models/deepfake_detector.pth')
print('Image forensics analyzer loaded successfully')
"

echo "Model training and testing completed successfully!"
