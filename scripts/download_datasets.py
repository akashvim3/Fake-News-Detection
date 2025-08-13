#!/usr/bin/env python3
"""
Script to download and prepare training datasets
"""
import os
import requests
import zipfile
import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Download and prepare datasets for training"""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def download_liar_dataset(self):
        """Download LIAR dataset for fake news detection"""
        logger.info("Downloading LIAR dataset...")
        
        urls = {
            "train": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv",
            "test": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv",
            "valid": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv"
        }
        
        for split, url in urls.items():
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                file_path = self.data_dir / f"liar_{split}.tsv"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                logger.info(f"Downloaded {split} split to {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to download {split} split: {e}")
    
    def download_fever_dataset(self):
        """Download FEVER dataset"""
        logger.info("Downloading FEVER dataset...")
        
        # Note: FEVER dataset is large, this is a simplified version
        urls = {
            "train": "https://fever.ai/download/fever/train.jsonl",
            "dev": "https://fever.ai/download/fever/shared_task_dev.jsonl"
        }
        
        for split, url in urls.items():
            try:
                logger.info(f"Downloading {split} split...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                file_path = self.data_dir / f"fever_{split}.jsonl"
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Downloaded {split} split to {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to download FEVER {split}: {e}")
                # Create sample data instead
                self._create_sample_fever_data(split)
    
    def _create_sample_fever_data(self, split: str):
        """Create sample FEVER data for testing"""
        sample_data = [
            {
                "id": 1,
                "claim": "The Earth is round and orbits the Sun.",
                "label": "SUPPORTS",
                "evidence": [["Earth", 0, "The Earth is the third planet from the Sun."]]
            },
            {
                "id": 2,
                "claim": "Vaccines cause autism in children.",
                "label": "REFUTES",
                "evidence": [["Vaccine", 0, "Multiple studies have found no link between vaccines and autism."]]
            },
            {
                "id": 3,
                "claim": "Climate change is caused by human activities.",
                "label": "SUPPORTS",
                "evidence": [["Climate change", 0, "Scientific consensus attributes climate change to human activities."]]
            }
        ]
        
        file_path = self.data_dir / f"fever_{split}.jsonl"
        with open(file_path, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Created sample FEVER data: {file_path}")
    
    def prepare_deepfake_dataset_info(self):
        """Prepare information about deepfake datasets"""
        info = {
            "datasets": {
                "FaceForensics++": {
                    "url": "https://github.com/ondyari/FaceForensics",
                    "description": "Large-scale deepfake detection dataset",
                    "size": "~500GB",
                    "note": "Requires registration and agreement to terms"
                },
                "Celeb-DF": {
                    "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
                    "description": "Celebrity deepfake dataset",
                    "size": "~15GB",
                    "note": "High-quality deepfakes"
                },
                "DFDC": {
                    "url": "https://deepfakedetectionchallenge.ai/",
                    "description": "Deepfake Detection Challenge dataset",
                    "size": "~470GB",
                    "note": "Kaggle competition dataset"
                }
            },
            "instructions": [
                "1. Register on the respective websites",
                "2. Agree to terms and conditions",
                "3. Download datasets to data/deepfakes/ directory",
                "4. Extract and organize files",
                "5. Run preprocessing scripts"
            ]
        }
        
        info_path = self.data_dir / "deepfake_datasets_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Deepfake dataset info saved to {info_path}")
    
    def create_sample_datasets(self):
        """Create sample datasets for testing"""
        logger.info("Creating sample datasets...")
        
        # Sample fake news data
        fake_news_data = {
            "texts": [
                "Breaking: Scientists discover vaccines contain microchips",
                "5G towers are spreading coronavirus worldwide",
                "The earth is flat according to new research",
                "Climate change is a hoax by scientists",
                "Moon landing was filmed in Hollywood studio"
            ],
            "labels": [1, 1, 1, 1, 1],  # All fake
            "sources": ["social_media"] * 5
        }
        
        real_news_data = {
            "texts": [
                "New AI breakthrough helps detect cancer early",
                "Renewable energy reaches 30% of global supply",
                "Archaeological discovery reveals ancient civilization",
                "Study shows exercise benefits for mental health",
                "Space telescope discovers new exoplanet"
            ],
            "labels": [0, 0, 0, 0, 0],  # All real
            "sources": ["news_outlet"] * 5
        }
        
        # Combine and save
        all_texts = fake_news_data["texts"] + real_news_data["texts"]
        all_labels = fake_news_data["labels"] + real_news_data["labels"]
        all_sources = fake_news_data["sources"] + real_news_data["sources"]
        
        df = pd.DataFrame({
            "text": all_texts,
            "label": all_labels,
            "source": all_sources
        })
        
        sample_path = self.data_dir / "sample_news_data.csv"
        df.to_csv(sample_path, index=False)
        
        logger.info(f"Sample dataset saved to {sample_path}")
    
    def download_all(self):
        """Download all available datasets"""
        logger.info("Starting dataset download process...")
        
        try:
            self.download_liar_dataset()
        except Exception as e:
            logger.error(f"LIAR download failed: {e}")
        
        try:
            self.download_fever_dataset()
        except Exception as e:
            logger.error(f"FEVER download failed: {e}")
        
        self.prepare_deepfake_dataset_info()
        self.create_sample_datasets()
        
        logger.info("Dataset preparation completed!")
        logger.info(f"Data directory: {self.data_dir.absolute()}")

def main():
    """Main function"""
    downloader = DatasetDownloader()
    downloader.download_all()

if __name__ == "__main__":
    main()
