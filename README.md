# FactCheck AI - Fake News & Deepfake Detection Tool

A cutting-edge, AI-powered platform designed to detect fake news and deepfakes using advanced machine learning techniques, combining state-of-the-art models like **BERT**, **OpenCV**, and **PyTorch**.

---

## 🚀 Key Features

* **Text Analysis**: BERT-based NLP for fake news detection.
* **Image Forensics**: Detect image manipulation and deepfakes.
* **Video Analysis**: Temporal consistency checks and deepfake detection.
* **Audio Forensics**: Identify synthetic speech and manipulated audio.
* **Multimodal Analysis**: Combine text, image, video, and audio insights.
* **REST API**: Seamless integration for developers.
* **Interactive Web Dashboard**: React-based, user-friendly analysis interface.
* **Real-time Processing**: Celery-based background job execution.
* **Scalable & Production Ready**: Docker, Kubernetes, monitoring included.

---

## 🛠️ Quick Start

### Prerequisites

* Python **3.8+**
* Node.js **16+** (for frontend)
* Redis Server
* PostgreSQL (optional; defaults to SQLite)

### Installation

1. **Clone the repository**

```bash
git clone <https://github.com/akashvim3/Fake-News-Detection.git/>
cd factcheck-api
```

2. **Run setup script**

```bash
chmod +x scripts/*.sh
./scripts/setup.sh
```

3. **Start development environment**

```bash
./scripts/start_dev.sh
```

4. **Access the application**

* API: `http://localhost:8000/api/v1/`
* Admin: `http://localhost:8000/admin/` (Credentials: `admin/admin123`)
* Health Check: `http://localhost:8000/api/v1/analysis/stats/`

---

## 📊 API Usage Examples

### Submit Analysis

```bash
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
```

### Check Status

```bash
curl http://localhost:8000/api/v1/analysis/{job_id}/status/
```

---

## 🧪 Testing

```bash
# Health check
python scripts/health_check.py

# Performance tests
python scripts/performance_test.py

# Run all tests
./scripts/run_all_checks.sh
```

---

## 🐳 Deployment

### Using Docker (Development)

```bash
docker-compose up -d
```

### Using Docker (Production)

```bash
docker-compose -f docker-compose.production.yml up -d
```

### Kubernetes Deployment

```bash
kubectl apply -f kubernetes/
```

---

## 📈 Monitoring

* Automated health checks
* Performance metrics (response time, throughput)
* Model evaluation statistics
* **Prometheus** for metrics
* **Grafana** dashboards

---

## 🔧 Configuration

Key files to modify:

* `.env` → Environment variables
* `factcheck_api/settings.py` → Django settings
* `docker-compose.yml` → Container setup
* `kubernetes/` → Kubernetes manifests

---

## 🛡️ Security

* Token-based authentication
* API rate limiting
* Input validation & sanitization
* SSL/TLS support
* CORS configuration
* Security headers

---

## 📚 API Endpoints Overview

* `POST /api/v1/analysis/analyze/` → Submit content for analysis
* `GET /api/v1/analysis/{id}/status/` → Check analysis status
* `POST /api/v1/analysis/{id}/feedback/` → Submit feedback
* `GET /api/v1/analysis/stats/` → Analysis statistics
* `POST /api/v1/auth/register/` → User registration
* `POST /api/v1/auth/login/` → User login

### Sample Response

```json
{
  "id": "uuid",
  "status": "completed",
  "verdict": "likely_misinformation",
  "confidence": 0.85,
  "summary": "Analysis summary",
  "evidence": ["source1", "source2"],
  "technical_appendix": {"details": "..."}
}
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add relevant tests
5. Open a pull request

---

## 📄 License

Licensed under the **MIT License**.

---

## 🆘 Support

* Run `python scripts/health_check.py` for quick diagnostics
* View logs: `tail -f logs/django.log`
* Run all checks: `./scripts/run_all_checks.sh`

---

## 🔄 Maintenance

### Backup Database

```bash
python scripts/backup_db.py
```

### Update Models

```bash
python scripts/train_models.sh
```

### Deploy Updates

```bash
./scripts/deploy_production.sh
```
