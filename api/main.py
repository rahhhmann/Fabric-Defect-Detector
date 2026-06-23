"""
Fabric Defect Detection API
FastAPI backend serving the trained YOLOv8m model.

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Docs available at:
    http://localhost:8000/docs
"""

import io
from pathlib import Path

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
IOU_THRESHOLD = 0.5  # lowered from Ultralytics default (0.7) to suppress
                      # true duplicate/overlapping boxes on the same defect

# Merge step config: handles same-class box fragmentation that NMS/iou
# alone cannot fix (e.g. a single long seam or warp/weft thread split
# into 2+ boxes with low mutual IoU). See api/postprocess.py for full
# rationale.
#
# "Warp_Weft" added alongside "seam": both are elongated, diagonal/
# line-shaped defects (a continuous thread run vs. a stitched seam),
# and both showed the same fragmentation pattern in practice — two
# overlapping same-class boxes on a single thread line (observed
# 2026-06-20: two Warp_Weft boxes at conf 0.28/0.45 on one diagonal
# thread, same failure mode as the original seam fragmentation case).
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

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Make sure best.pt is placed inside the models/ folder."
    )

model = YOLO(MODEL_PATH)


def read_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into an OpenCV BGR image."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_merged_boxes(img: np.ndarray, defects: list[dict]) -> np.ndarray:
    """
    Draw the MERGED defect boxes (post-postprocessing) onto a copy
    of the original image using OpenCV.

    This exists because Ultralytics' `results.plot()` only knows
    about the raw, pre-merge boxes (it has no awareness of our
    custom merge_fragmented_detections() step). To keep the
    annotated PNG visually consistent with what /predict reports
    as JSON (same count, same boxes), we draw the merged boxes
    ourselves instead of relying on results.plot().
    """
    annotated = img.copy()

    # Fixed BGR color per class for visual consistency across requests.
    class_colors = {
        "Stain": (0, 165, 255),      # orange
        "Thread": (255, 0, 0),       # blue
        "Warp_Weft": (0, 255, 255),  # yellow
        "hole": (0, 0, 255),         # red
        "seam": (255, 0, 255),       # magenta
    }
    default_color = (0, 255, 0)  # green fallback for any unlisted class

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
    """
    Run YOLO inference, apply the fragment-merge postprocessing
    step, and return the final merged defect list.

    The returned `defects` list is the single source of truth for
    ALL endpoints — /predict, /predict/batch, AND /predict/annotated
    (via draw_merged_boxes()) — so JSON output and the annotated
    image always agree on count and box positions.
    """
    results = model.predict(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]

    defects = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        defects.append({
            "type": model.names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],  # [x1, y1, x2, y2]
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
    return {"status": "ok", "model_loaded": True, "classes": list(model.names.values())}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Return JSON detection results for a single image."""
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
    """
    Return the image with MERGED detection boxes drawn on it (PNG).

    Uses the same post-merge `defects` list as /predict, so the
    box count and positions you see here always match the JSON
    response exactly — no raw/fragmented duplicate boxes.
    """
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
    """Run detection on multiple images at once — for factory QC batch mode."""
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