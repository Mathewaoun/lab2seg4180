# Submission Guide - House Segmentation Service

## ✅ Project Status: READY TO SUBMIT

All tests pass and the project is fully functional!

---

## 📋 What You Have

### **Project: House Segmentation Service (Lab 2)**

A Flask-based API for aerial image house segmentation with:

- **API Server**: Flask application serving predictions via HTTP
- **Model**: PyTorch U-Net for binary image segmentation
- **Authentication**: API key protection (configurable)
- **Docker Support**: Containerized deployment ready
- **Testing**: 10 automated tests (all passing ✅)
- **CI/CD**: GitHub Actions workflow for testing & Docker publishing
- **Configuration**: Environment-based settings via `.env`

---

## 🧪 Running Tests Locally

### **Prerequisites**
Virtual environment created with all dependencies installed.

### **Run All Tests**
```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

**Expected Output**: ✅ 10/10 tests passed

### **Run Specific Test File**
```bash
python -m pytest tests/test_api.py -v
```

### **Test Coverage**
- ✅ Health endpoint reporting
- ✅ API key authentication
- ✅ File upload validation
- ✅ Mask payload generation
- ✅ Metrics computation
- ✅ Dataset preparation
- ✅ SAM mask precision

---

## 🚀 Running the API

### **Start the Server**
```bash
source .venv/bin/activate
python app.py
```

**Server starts at**: `http://0.0.0.0:5001` (configured in `.env`)

### **Test the API**

**1. Health Check:**
```bash
curl http://localhost:5001/health
```

**2. Make a Prediction:**
```bash
curl -X POST http://localhost:5001/predict \
  -F "file=@path/to/image.png" \
  -H "X-API-Key: lab2-secret"
```

**3. Prediction with Ground Truth:**
```bash
curl -X POST http://localhost:5001/predict \
  -F "file=@path/to/image.png" \
  -F "ground_truth=@path/to/mask.png" \
  -H "X-API-Key: lab2-secret"
```

---

## 🐳 Docker Deployment

### **Build Docker Image**
```bash
docker build -t house-segmentation-service:latest .
```

### **Run Docker Container**
```bash
docker run -p 5001:5000 \
  -e MODEL_WEIGHTS_PATH=artifacts_final/checkpoints/best_model.pt \
  -e REQUIRE_API_KEY=true \
  -e MODEL_SERVICE_API_KEY=lab2-secret \
  house-segmentation-service:latest
```

---

## 📤 Push to GitHub

### **1. Create a Repository on GitHub**
- Go to https://github.com/new
- Create a new repository (e.g., `house-segmentation-service`)
- **Do NOT** initialize with README (you already have one)

### **2. Add Remote and Push**
```bash
cd /Users/matto/Downloads/model-service

# Add remote (replace USERNAME and REPO-NAME)
git remote add origin https://github.com/USERNAME/REPO-NAME.git

# Rename branch to main if needed
git branch -M main

# Push all commits
git push -u origin main
```

### **3. GitHub Actions CI/CD**
Once pushed, GitHub Actions will automatically:
1. ✅ Run all tests
2. ✅ Build Docker image
3. ✅ Optionally publish to Docker Hub (requires secrets)

**View Results**: Go to "Actions" tab in your GitHub repo

---

## 📁 Project Structure

```
model-service/
├── app.py                          # Flask API entry point
├── Dockerfile                      # Docker configuration
├── requirements.txt                # Python dependencies
├── .env                            # Environment config (git-ignored)
├── .env.example                    # Example config template
├── README.md                        # Project documentation
│
├── model_service/                  # Core package
│   ├── config.py                   # Settings management
│   ├── inference.py                # Prediction logic
│   ├── unet.py                     # U-Net model architecture
│   ├── training.py                 # Training utilities
│   ├── metrics.py                  # IoU, Dice computation
│   └── data.py                     # Data loading utilities
│
├── scripts/                        # Training & evaluation
│   ├── prepare_dataset.py          # Data preparation (local/SAM modes)
│   ├── train_segmentation.py       # Model training
│   └── evaluate_segmentation.py    # Performance evaluation
│
├── tests/                          # Automated tests
│   ├── test_api.py                 # API endpoint tests
│   ├── test_metrics.py             # Metrics tests
│   ├── test_prepare_dataset.py     # Dataset tests
│   └── test_sam_precision.py       # SAM precision tests
│
├── artifacts_final/                # Pre-trained model weights
│   └── checkpoints/best_model.pt   # U-Net weights
│
├── data/processed_final/           # Processed dataset
│   ├── train/                      # Training samples
│   ├── val/                        # Validation samples
│   └── test/                       # Test samples
│
└── .github/
    └── workflows/
        └── ci-cd.yml               # GitHub Actions pipeline
```

---

## 🔧 Configuration

Edit `.env` to customize:

```properties
APP_HOST=0.0.0.0              # API listen address
APP_PORT=5001                 # API port
MODEL_WEIGHTS_PATH=...        # Path to model checkpoint
PREDICTION_SIZE=256           # Resize images to this size
PREDICTION_THRESHOLD=0.5      # Binary classification threshold
MODEL_DEVICE=cpu              # 'cpu', 'cuda', or 'auto'
REQUIRE_API_KEY=true          # Enforce X-API-Key header
MODEL_SERVICE_API_KEY=...     # Secret API key
DATA_DIR=data/processed_final # Dataset location
ARTIFACTS_DIR=artifacts_final # Model artifacts location
```

---

## 📊 Key Features

### **API Endpoints**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Service info |
| `/health` | GET | Model status & health check |
| `/predict` | POST | Generate segmentation mask |

### **Security**

- Optional API key authentication via `X-API-Key` header
- Configurable in `.env` with `REQUIRE_API_KEY` and `MODEL_SERVICE_API_KEY`

### **Model Output**

Returns JSON with:
- `mask_png_base64` - Base64-encoded PNG prediction mask
- `foreground_ratio` - Percentage of image predicted as house
- `dice_score`, `iou_score` (optional) - Metrics if ground truth provided
- `message` - Status message

---

## 📝 Pre-Submission Checklist

- [x] All 10 tests pass ✅
- [x] API starts without errors ✅
- [x] Environment configuration working ✅
- [x] Docker builds successfully ✅
- [x] Git repository initialized ✅
- [x] All files committed ✅
- [x] README documentation complete ✅
- [x] CI/CD workflow configured ✅

---

## 🚨 Troubleshooting

### **Port Already in Use**
```bash
# Kill the process on port 5001
lsof -i :5001
kill -9 <PID>
```

### **Module Import Errors**
```bash
# Ensure venv is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### **Model Weights Not Found**
Check that `artifacts_final/checkpoints/best_model.pt` exists and path is correct in `.env`

---

## 📬 Ready to Submit!

Your project is fully functional and ready for submission. Simply:

1. **Push to GitHub** (see steps above)
2. **GitHub Actions will automatically test it**
3. **Share the repository link**

Good luck! 🎉
