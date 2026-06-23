"""
Fabric Defect Detection API
FastAPI backend serving the trained YOLOv8m model.
"""

import io
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from ultralytics import YOLO

from api.postprocess import PostprocessConfig, merge_fragmented_detections

# ── Config ──────────────────────────────────────────────
MODEL_PATH = "models/best.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

POSTPROCESS_CONFIG = PostprocessConfig(
    merge_classes={"seam", "Warp_Weft"},
    gap_threshold_px=20.0,
    elongation_ratio_min=1.5,
    max_angle_diff_deg=35.0,
)
# ────────────────────────────────────────────────────────

app = FastAPI(
    title="Fabric Defect Detection API",
    description="Detects Stain, Thread, Warp_Weft, hole, and seam defects in fabric images.",
    version="1.1.0",
)

# ── Lazy model loader ────────────────────────────────────
@lru_cache(maxsize=1)
def get_model() -> YOLO:
    """Load model once on first request, cache for subsequent calls."""
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Make sure best.pt is placed inside the models/ folder."
        )
    return YOLO(MODEL_PATH)
# ────────────────────────────────────────────────────────


def read_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_merged_boxes(img: np.ndarray, defects: list[dict]) -> np.ndarray:
    annotated = img.copy()
    class_colors = {
        "Stain": (0, 165, 255),
        "Thread": (255, 0, 0),
        "Warp_Weft": (0, 255, 255),
        "hole": (0, 0, 255),
        "seam": (255, 0, 255),
    }
    default_color = (0, 255, 0)

    for defect in defects:
        x1, y1, x2, y2 = [int(round(v)) for v in defect["bbox"]]
        color = class_colors.get(defect["type"], default_color)
        label = f'{defect["type"]} {defect["confidence"]:.2f}'

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y1 = max(0, y1 - text_h - baseline - 4)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated


def run_inference(img: np.ndarray):
    model = get_model()  # lazy load
    results = model.predict(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]

    defects = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        defects.append({
            "type": model.names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
        })

    defects = merge_fragmented_detections(defects, POSTPROCESS_CONFIG)
    return defects


@app.get("/")
def root():
    return {
        "message": "Fabric Defect Detection API is running",
        "endpoints": ["/predict", "/predict/annotated", "/predict/batch", "/health"],
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": Path(MODEL_PATH).exists(), "classes": ["Stain", "Thread", "Warp_Weft", "hole", "seam"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    img_bytes = await file.read()
    img = read_image(img_bytes)
    defects = run_inference(img)
    return {
        "filename": file.filename,
        "defects_found": len(defects) > 0,
        "count": len(defects),
        "defects": defects,
        "verdict": "REJECT" if defects else "PASS",
    }


@app.post("/predict/annotated")
async def predict_annotated(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    img_bytes = await file.read()
    img = read_image(img_bytes)
    defects = run_inference(img)
    annotated = draw_merged_boxes(img, defects)
    success, encoded = cv2.imencode(".png", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
    return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/png")


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    results_list = []
    passed = 0
    rejected = 0

    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        img_bytes = await file.read()
        img = read_image(img_bytes)
        defects = run_inference(img)
        verdict = "REJECT" if defects else "PASS"
        if verdict == "PASS":
            passed += 1
        else:
            rejected += 1
        results_list.append({
            "filename": file.filename,
            "verdict": verdict,
            "count": len(defects),
            "defects": defects,
        })

    total = len(results_list)
    pass_rate = f"{(passed / total) * 100:.1f}%" if total else "0.0%"
    return {
        "summary": {
            "total_checked": total,
            "passed": passed,
            "rejected": rejected,
            "pass_rate": pass_rate,
        },
        "results": results_list,
    }