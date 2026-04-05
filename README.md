# AI Powered Pharmaceutical Supply Chain Integrity and Optimization System

A Flask-based AI pharmacy dashboard for medicine authenticity verification using deep learning, OCR, scan history, and future-ready barcode integration.

This project combines three trained image classification models with OCR-based text validation to predict whether a medicine package is likely `REAL` or `FAKE`. It also provides a clean web dashboard for scanning, reviewing results, and maintaining historical records for demo and research use.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [How the Prediction Pipeline Works](#how-the-prediction-pipeline-works)
- [Installation Guide](#installation-guide)
- [How to Run the Application](#how-to-run-the-application)
- [Application Pages](#application-pages)
- [API Endpoints](#api-endpoints)
- [Database Schema](#database-schema)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Troubleshooting](#troubleshooting)
- [Author](#author)

## Project Overview

The **AI Powered Pharmaceutical Supply Chain Integrity and Optimization System** is an AI-assisted medicine verification platform designed for healthcare demos, academic research, and M.Tech project presentations.

The system uses:

- `ResNet50`
- `EfficientNet-B4`
- `ViT Base`
- `Tesseract OCR`
- an ensemble-based decision pipeline

to classify medicine package images as authentic or suspicious.

Along with prediction, the application provides:

- a modern SaaS-style dashboard
- image upload and camera capture
- OCR text extraction display
- detailed model output scores
- SQLite-based scan history
- barcode/QR capture support for future supply-chain extensions

## Key Features

- Dashboard with real-time scan statistics
- Upload medicine image for verification
- Live camera capture using browser `getUserMedia`
- OCR text extraction from medicine packaging
- Final ensemble prediction: `REAL` or `FAKE`
- Confidence score display
- Detailed debugging output:
  - ResNet score
  - EfficientNet score
  - ViT score
  - OCR score
  - Final ensemble score
- Result image preview
- SQLite scan history with timestamps
- History page with searchable visual record format
- Barcode / QR scan capture endpoint
- Future-ready structure for blockchain or supply-chain verification integration
- Responsive, presentation-ready Flask UI

## Technology Stack

### Backend

- Flask
- Python
- SQLite

### AI / ML

- PyTorch
- TorchVision
- TIMM
- Scikit-learn

### Computer Vision / OCR

- OpenCV
- PyTesseract
- Tesseract OCR Engine

### Frontend

- HTML
- CSS
- Minimal JavaScript
- `html5-qrcode` for barcode/QR scanning in browser

## System Requirements

### Minimum

- OS: Windows 10/11, Linux, or macOS
- Python: `3.10+` recommended
- RAM: `8 GB` minimum
- Storage: `2 GB+` free space excluding dataset growth
- Webcam: required for camera capture and barcode scanning
- Browser: latest Chrome / Edge recommended

### Recommended

- Python `3.10` or `3.11`
- RAM: `16 GB`
- NVIDIA GPU with CUDA support for faster inference
- SSD storage

### External Dependencies

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) must be installed on the system
- Trained model weight files must exist inside the [`models`](./models) folder:
  - `models/resnet50.pth`
  - `models/efficientnet.pth`
  - `models/vit.pth`

## Project Structure

```text
ML Models/
├── app.py
├── main.py
├── requirements.txt
├── models/
│   ├── resnet50.pth
│   ├── efficientnet.pth
│   └── vit.pth
├── utils/
│   ├── inference.py
│   ├── ocr.py
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   └── ensemble.py
├── templates/
│   ├── index.html
│   ├── scan.html
│   └── history.html
├── static/
│   ├── style.css
│   ├── dashboard.js
│   ├── scan.js
│   ├── sidebar.js
│   └── uploads/
├── pharmacy_dashboard/
│   ├── __init__.py
│   ├── routes.py
│   └── services/
└── instance/
    └── scan_history.db
```

## How the Prediction Pipeline Works

1. User uploads or captures a medicine package image.
2. The image is saved under `static/uploads/`.
3. The system loads the trained models:
   - ResNet50
   - EfficientNet-B4
   - ViT Base
4. Each model predicts a probability score.
5. OCR extracts text from the medicine package.
6. OCR text is validated using rule-based checks.
7. Final ensemble score is computed:

```text
Final Score = (0.7 × Image Model Average) + (0.3 × OCR Score)
```

8. The system returns:
   - `REAL` or `FAKE`
   - confidence score
   - OCR text
   - detailed model outputs
9. The result is saved automatically to SQLite history.

## Installation Guide

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd "ML Models"
```

### 2. Create a Virtual Environment

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

#### Windows

1. Download and install Tesseract OCR.
2. During installation, note the install path.
3. Add the Tesseract installation folder to your system `PATH`.

Common path:

```text
C:\Program Files\Tesseract-OCR\
```

#### Ubuntu / Debian

```bash
sudo apt update
sudo apt install tesseract-ocr
```

#### macOS

```bash
brew install tesseract
```

### 5. Verify Trained Model Files

Ensure the following files already exist:

```text
models/resnet50.pth
models/efficientnet.pth
models/vit.pth
```

If these files are missing, the Flask app will not start prediction correctly.

### 6. Optional: GPU Support

If using NVIDIA GPU, install the CUDA-compatible PyTorch build from the official PyTorch website:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## How to Run the Application

Start the Flask app:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## Application Pages

### 1. Dashboard

Shows:

- Total scans
- Real medicines count
- Fake medicines count
- Overview of the project and verification workflow

### 2. Scan Medicine

Supports:

- image upload
- camera capture
- barcode / QR scanning

Displays:

- final prediction
- confidence score
- image preview
- OCR extracted text
- detailed model scores

### 3. History

Shows:

- previously scanned records
- result labels
- confidence
- image reference
- timestamp

## API Endpoints

### `POST /api/predict`

Accepts:

- uploaded image file
- or captured base64 image

Returns:

```json
{
  "ok": true,
  "prediction": "REAL",
  "confidence": 0.82,
  "ocr_text": "Extracted text",
  "score": 0.82,
  "detailed_scores": {
    "resnet_score": 0.81,
    "efficientnet_score": 0.80,
    "vit_score": 0.84,
    "ocr_score": 0.66,
    "ensemble_score": 0.82
  },
  "image_url": "/static/uploads/sample.jpg",
  "history_record": {
    "id": 1
  }
}
```

### `POST /scan-barcode`

Accepts:

```json
{
  "barcode_value": "1234567890"
}
```

Returns:

```json
{
  "ok": true,
  "barcode": {
    "barcode_value": "1234567890",
    "status": "captured",
    "blockchain_ready": true,
    "verification_stage": "pending_future_blockchain_integration"
  }
}
```

### `GET /history`

Renders the scan history page.

## Database Schema

The application creates an SQLite database automatically at:

```text
instance/scan_history.db
```

Table: `scan_history`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key |
| `image_path` | TEXT | Saved image path |
| `result` | TEXT | `REAL` or `FAKE` |
| `confidence` | REAL | Confidence score |
| `extracted_text` | TEXT | OCR output |
| `timestamp` | TEXT | Scan timestamp |

## Future Enhancements

- Blockchain-backed pharmaceutical verification
- Product batch lookup
- Manufacturer database integration
- Barcode validation against centralized medicine registry
- User authentication and role-based access
- Export reports as PDF / CSV
- Cloud deployment
- Real-time analytics dashboard

## Troubleshooting

### Models not loading

Check that:

- all `.pth` model files are present in the `models/` folder
- model file names exactly match the expected names

### OCR not working

Check that:

- Tesseract OCR is installed
- Tesseract is added to system `PATH`
- uploaded image quality is clear enough for text extraction

### Camera not opening

Check that:

- browser camera permission is enabled
- you are using a modern browser
- no other app is blocking the webcam

### Barcode scanner not starting

Check that:

- browser camera permission is allowed
- the page is fully loaded
- you are using Chrome or Edge for best compatibility

## Author

**Darshan Taskar**

Developed as an AI-powered pharmaceutical verification dashboard for academic demonstration and healthcare technology research.
