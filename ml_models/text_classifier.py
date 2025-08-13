"""
Real BERT-based text classifier for fake news detection
"""
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class FakeNewsClassifier:
    """BERT-based fake news classifier"""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.classifier = None
        
    def load_model(self, model_path: str = None):
        """Load pre-trained model or initialize new one"""
        try:
            if model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, 
                    num_labels=self.num_labels
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, 
                    num_labels=self.num_labels
                )
            
            # Create pipeline for inference
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=True
            )
            
            logger.info(f"Model loaded successfully from {model_path or self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (BERT has 512 token limit)
        if len(text) > 2000:  # Rough character limit
            text = text[:2000] + "..."
        
        return text
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict if text is fake news"""
        if not self.classifier:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Get predictions
            results = self.classifier(processed_text)
            
            # Extract confidence scores
            fake_score = next(r['score'] for r in results if r['label'] == 'LABEL_1')
            real_score = next(r['score'] for r in results if r['label'] == 'LABEL_0')
            
            # Determine verdict
            if fake_score > 0.7:
                verdict = 'likely_misinformation'
            elif real_score > 0.7:
                verdict = 'likely_true'
            else:
                verdict = 'inconclusive'
            
            confidence = max(fake_score, real_score)
            
            return {
                'verdict': verdict,
                'confidence': confidence,
                'fake_probability': fake_score,
                'real_probability': real_score,
                'raw_results': results
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'verdict': 'inconclusive',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def extract_claims(self, text: str) -> List[str]:
        """Extract individual claims from text using sentence segmentation"""
        import nltk
        try:
            nltk.download('punkt', quiet=True)
            from nltk.tokenize import sent_tokenize
            
            sentences = sent_tokenize(text)
            
            # Filter out very short sentences
            claims = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            return claims[:5]  # Return max 5 claims
            
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            # Fallback to simple splitting
            sentences = text.split('.')
            return [s.strip() for s in sentences if len(s.strip()) > 20][:5]

class FakeNewsTrainer:
    """Training pipeline for fake news classifier"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
    
    def prepare_dataset(self, texts: List[str], labels: List[int], max_length: int = 512):
        """Tokenize texts for training"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        class NewsDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
            
            def __len__(self):
                return len(self.labels)
        
        return NewsDataset(encodings, labels)
    
    def train(self, train_texts: List[str], train_labels: List[int], 
              val_texts: List[str], val_labels: List[int],
              output_dir: str = "./fake_news_model",
              num_epochs: int = 3,
              batch_size: int = 16):
        """Train the fake news classifier"""
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        val_dataset = self.prepare_dataset(val_texts, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
        )
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
            accuracy = accuracy_score(labels, predictions)
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
        
        return trainer

def load_training_data() -> Tuple[List[str], List[int]]:
    """Load training data from various sources"""
    # This would load from actual datasets like LIAR, FEVER, etc.
    # For demo purposes, using sample data
    
    fake_news_samples = [
        "Scientists have discovered that vaccines contain microchips for government surveillance.",
        "Breaking: COVID-19 was created in a laboratory to control population.",
        "New study shows that 5G towers cause cancer and spread viruses.",
        "Doctors don't want you to know this simple trick that cures all diseases.",
        "Government is hiding the truth about flat earth from the public."
    ]
    
    real_news_samples = [
        "Researchers at MIT have developed a new method for detecting deepfakes using AI.",
        "The World Health Organization recommends vaccination as the best protection against COVID-19.",
        "Climate scientists report record-breaking temperatures in the Arctic region.",
        "New archaeological discovery sheds light on ancient civilizations.",
        "Economic indicators show steady growth in the technology sector."
    ]
    
    texts = fake_news_samples + real_news_samples
    labels = [1] * len(fake_news_samples) + [0] * len(real_news_samples)  # 1 = fake, 0 = real
    
    return texts, labels

if __name__ == "__main__":
    # Example usage
    classifier = FakeNewsClassifier()
    classifier.load_model()
    
    test_text = "Breaking: Scientists discover that vaccines contain microchips for mind control."
    result = classifier.predict(test_text)
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.2f}")
