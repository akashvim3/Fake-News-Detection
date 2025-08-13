# FactCheck AI - Fake News & Deepfake Detection Tool

A comprehensive AI-powered platform for detecting fake news and deepfakes using advanced machine learning models including BERT, OpenCV, and PyTorch.

## 🚀 Features

- **Text Analysis**: BERT-based fake news detection
- **Image Forensics**: Deepfake and manipulation detection
- **Video Analysis**: Temporal consistency and deepfake detection
- **Audio Forensics**: Synthetic speech detection
- **Multimodal Analysis**: Combined text and media analysis
- **REST API**: Complete API for integration
- **Web Dashboard**: React-based analysis interface
- **Real-time Processing**: Celery-based background processing
- **Production Ready**: Docker, Kubernetes, monitoring

## 🛠️ Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend)
- Redis Server
- PostgreSQL (optional, SQLite by default)

### Installation

1. **Clone the repository**
\`\`\`bash
git clone <repository-url>
cd factcheck-api
\`\`\`

2. **Run setup script**
\`\`\`bash
chmod +x scripts/*.sh
./scripts/setup.sh
\`\`\`

3. **Start development environment**
\`\`\`bash
./scripts/start_dev.sh
\`\`\`

4. **Access the application**
- API: http://localhost:8000/api/v1/
- Admin: http://localhost:8000/admin/ (admin/admin123)
- Health Check: http://localhost:8000/api/v1/analysis/stats/

## 📊 API Usage

### Submit Analysis

\`\`\`bash
# Text analysis
curl -X POST http://localhost:8000/api/v1/analysis/analyze/ \
  -H "Content-Type: application/json" \
  -d '{
    "content_type": "text",
    "text_content": "Breaking news: Scientists discover vaccines contain microchips"
  }'

# Image analysis
curl -X POST http://localhost:8000/api/v1/analysis/analyze/ \
  -F "content_type=image" \
  -F "file_upload=@image.jpg"
\`\`\`

### Check Status

\`\`\`bash
curl http://localhost:8000/api/v1/analysis/{job_id}/status/
\`\`\`

## 🧪 Testing

\`\`\`bash
# Run health check
python scripts/health_check.py

# Run performance tests
python scripts/performance_test.py

# Run all tests
./scripts/run_all_checks.sh
\`\`\`

## 🐳 Docker Deployment

### Development
\`\`\`bash
docker-compose up -d
\`\`\`

### Production
\`\`\`bash
docker-compose -f docker-compose.production.yml up -d
\`\`\`

## ☸️ Kubernetes Deployment

\`\`\`bash
kubectl apply -f kubernetes/
\`\`\`

## 📈 Monitoring

- **Health Checks**: Automated system health monitoring
- **Performance Metrics**: Response time and throughput tracking
- **Model Evaluation**: Accuracy and performance metrics
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards

## 🔧 Configuration

Key configuration files:
- `.env`: Environment variables
- `factcheck_api/settings.py`: Django settings
- `docker-compose.yml`: Container orchestration
- `kubernetes/`: Kubernetes manifests

## 🛡️ Security

- Token-based authentication
- Rate limiting
- Input validation
- SSL/TLS support
- CORS configuration
- Security headers

## 📚 API Documentation

### Endpoints

- `POST /api/v1/analysis/analyze/` - Submit content for analysis
- `GET /api/v1/analysis/{id}/status/` - Get analysis results
- `POST /api/v1/analysis/{id}/feedback/` - Submit user feedback
- `GET /api/v1/analysis/stats/` - Get analysis statistics
- `POST /api/v1/auth/register/` - User registration
- `POST /api/v1/auth/login/` - User login

### Response Format

\`\`\`json
{
  "id": "uuid",
  "status": "completed",
  "verdict": "likely_misinformation",
  "confidence": 0.85,
  "summary": "Analysis summary",
  "evidence": [...],
  "technical_appendix": {...}
}
\`\`\`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Check the health status: `python scripts/health_check.py`
- View logs: `tail -f logs/django.log`
- Run diagnostics: `./scripts/run_all_checks.sh`

## 🔄 Maintenance

### Backup Database
\`\`\`bash
python scripts/backup_db.py
\`\`\`

### Update Models
\`\`\`bash
python scripts/train_models.sh
\`\`\`

### Deploy Updates
\`\`\`bash
./scripts/deploy_production.sh
