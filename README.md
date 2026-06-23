# Fabric Defect Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8m-Ultralytics-purple?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=for-the-badge)

An end-to-end automated fabric defect detection system for Bangladesh's RMG industry.  
Detects 5 defect classes in real time using YOLOv8m, served via a production-grade FastAPI backend and an interactive Streamlit QC dashboard — fully containerized with Docker.

[Live Demo](#deployment) · [API Docs](#api-reference) · [Model Results](#model-performance)

</div>

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Defect Classes](#defect-classes)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Results](#results)
- [Author](#author)

---

## Problem Statement

Bangladesh is the world's second-largest garment exporter, contributing over **$47 billion** in annual exports. Manual fabric quality inspection remains a major bottleneck — it is slow, inconsistent, and difficult to scale.

This project delivers an AI-powered automated QC system that detects fabric defects in real time using computer vision, enabling RMG factories to reduce waste, improve throughput, and maintain export quality standards.

---

## Key Features

- 5-class real-time defect detection: Stain, Thread, Warp/Weft, Hole, Seam
- YOLOv8m (medium variant) achieving **mAP50 = 0.8425** on the held-out test set
- REST API with three endpoints: single predict, annotated image, batch QC
- Interactive Streamlit dashboard with per-class confidence breakdown
- Fully Dockerized — single command to run the entire platform locally
- Deployed on Render with separate API and Dashboard services
- Two training runs with seam class oversampling in v2 to address class imbalance

---

## System Architecture

```mermaid
flowchart TD
    subgraph CLIENT["Client"]
        UI["Streamlit Dashboard\nfabric-qc-dashboard.onrender.com"]
        SWAGGER["Swagger UI\n/docs"]
    end

    subgraph API_SERVICE["Docker Container — API Service"]
        FW["FastAPI\nport 8000"]
        MODEL["YOLOv8m\nbest.pt · mAP50: 0.8425"]
        subgraph ENDPOINTS["Endpoints"]
            E1["/predict\nJSON detections"]
            E2["/predict/annotated\nAnnotated image"]
            E3["/predict/batch\nBatch QC summary"]
        end
        FW --> ENDPOINTS
        ENDPOINTS --> MODEL
    end

    subgraph APP_SERVICE["Docker Container — App Service"]
        ST["Streamlit\nport 8501"]
    end

    subgraph TRAINING["Training Pipeline — Kaggle T4 GPU"]
        DS["Dataset\nRoboflow Universe"]
        RUN1["Run 1 · YOLOv8m v1\nmAP50: 0.7891"]
        RUN2["Run 2 · YOLOv8m v2\nSeam Oversampled\nmAP50: 0.8425"]
        DS --> RUN1 --> RUN2
        RUN2 -->|best.pt| MODEL
    end

    UI -->|"HTTP POST · API_URL env var"| FW
    SWAGGER --> FW
    ST --> UI
```

---

## Project Structure

```
fabric-defect-detector/
│
├── api/
│   ├── main.py                          # FastAPI entrypoint — 3 endpoints
│   └── requirements.txt
│
├── app/
│   ├── dashboard.py                     # Streamlit QC dashboard
│   ├── requirements.txt
│   └── .streamlit/
│       └── config.toml
│
├── data/
│   ├── raw/                             # Original Roboflow dataset
│   └── processed/                       # Train/val/test splits (YOLO format)
│
├── models/
│   └── best.pt                          # YOLOv8m v2 weights (52 MB)
│
├── notebooks/
│   ├── 01_training_v1.ipynb
│   └── 02_training_v2_oversampled.ipynb
│
├── Results/
│   ├── Run-1/                           # v1 metrics, curves, confusion matrix
│   └── Run-2-over-sampled/              # v2 metrics + oversample_seam.py
│
├── sample_images/
├── src/
│
├── Dockerfile.api
├── Dockerfile.app
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── requirements.txt
```

---

## Defect Classes

| Class | Description |
|-------|-------------|
| Stain | Oil, chemical, or dirt contamination on fabric surface |
| Thread | Loose or broken thread visible on fabric |
| Warp_Weft | Structural weaving defects (warp/weft errors) |
| Hole | Physical holes or tears in fabric |
| Seam | Seam-related defects (underrepresented in v1; oversampled in v2) |

Seam class suffered from severe class imbalance in v1. Run 2 applied targeted oversampling via `oversample_seam.py`, which was the primary driver of the +5.34% mAP50 improvement.

---

## Model Performance

### Run Comparison

| Metric | Run 1 — v1 Baseline | Run 2 — v2 Oversampled |
|--------|---------------------|------------------------|
| mAP50 | 0.7891 | **0.8425** |
| Model | YOLOv8m | YOLOv8m |
| Seam Recall | Low | Improved |
| Epochs | 50 | 70 |
| Hardware | Kaggle T4 GPU | Kaggle T4 GPU |

### v2 Approximate Per-Class mAP50

| Class | mAP50 |
|-------|-------|
| Stain | ~0.89 |
| Thread | ~0.86 |
| Warp_Weft | ~0.84 |
| Hole | ~0.81 |
| Seam | ~0.78 |

Full metrics, precision-recall curves, and confusion matrices are in [`Results/Run-2-over-sampled/`](./Results/Run-2-over-sampled/).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | YOLOv8m (Ultralytics) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |
| Training | Kaggle (T4 GPU) |
| Dataset | Roboflow Universe |
| Deployment | Render |
| Language | Python 3.10 |

---

## Quick Start

### Docker Compose (Recommended)

```bash
git clone https://github.com/rahhhmann/Fabric-Defect-Detector.git
cd Fabric-Defect-Detector
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Backend | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |

### Manual Setup

```bash
# Terminal 1 — API
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 — Dashboard
cd app
pip install -r requirements.txt
API_URL=http://localhost:8000 streamlit run dashboard.py
```

---

## API Reference

### `POST /predict`

Returns JSON with bounding boxes, class labels, and confidence scores.

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample_images/test.jpg"
```

```json
{
  "detections": [
    {
      "class": "Stain",
      "confidence": 0.91,
      "bbox": [120, 85, 340, 210]
    }
  ],
  "total_defects": 1,
  "inference_time_ms": 47.3
}
```

### `POST /predict/annotated`

Returns the input image with bounding boxes rendered — suitable for direct display.

```bash
curl -X POST "http://localhost:8000/predict/annotated" \
  -F "file=@sample_images/test.jpg" \
  --output annotated.jpg
```

### `POST /predict/batch`

Accepts multiple images, returns per-image defect counts and an aggregate QC summary.

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg"
```

Full interactive documentation is available at `/docs` (Swagger UI).

---

## Deployment

Deployed as two independent Web Services on Render:

| Service | URL |
|---------|-----|
| FastAPI API | `https://fabric-qc-api.onrender.com` |
| Streamlit Dashboard | `https://fabric-qc-dashboard.onrender.com` |

### Deploy Your Own

1. Fork this repository
2. Go to [render.com](https://render.com) → New Web Service
3. **API service:** Runtime = Docker, Dockerfile Path = `Dockerfile.api`
4. **Dashboard service:** Runtime = Docker, Dockerfile Path = `Dockerfile.app`
5. On the Dashboard service, set environment variable: `API_URL=https://<your-api-name>.onrender.com`

> Free tier services spin down after 15 minutes of inactivity. The first request after idle may take 30–60 seconds to respond (cold start).

---

## Results

Training artifacts are versioned under the `Results/` directory:

- `Results/Run-1/` — v1 baseline: metrics CSV, training curves, confusion matrix
- `Results/Run-2-over-sampled/` — v2 final: metrics CSV, PR curves, `oversample_seam.py`

---

## Author

**Ashikur Rahman**  
Final-year CSE, Patuakhali Science and Technology University  
GitHub: [rahhhmann](https://github.com/rahhhmann) · HuggingFace: [ashik297](https://huggingface.co/ashik297)

---

## License

This project is licensed under the [MIT License](LICENSE).
