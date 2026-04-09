"""
╔══════════════════════════════════════════════════════════════╗
║          AGROSHIELD — FastAPI Backend (app.py)              ║
║   ConvNeXt-Tiny · Grad-CAM · Upload + ESP-32 IP Camera      ║
╚══════════════════════════════════════════════════════════════╝
Run:  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import base64
import csv
import os

import cv2
import numpy as np
import requests
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image


# ─────────────────────────────────────────────────────────────
# APP INIT + CORS
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agroshield Plant Disease API",
    description="ConvNeXt-Tiny inference with Grad-CAM visualisation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins (local HTML, ESP-32, etc.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
MODEL_PATH     = "plant_model_complete/convnext_plant_disease.keras"
CLASS_CSV_PATH = "class_dict.csv"
IMG_SIZE       = (224, 224)

# ConvNeXt last convolutional block name for Grad-CAM
# Adjust if your saved model uses a different name (check model.summary())
GRADCAM_LAYER  = "convnext_tiny"

# ── ESP-32 camera URL ────────────────────────────────────────
# Set the ESP32_CAPTURE_URL environment variable to override.
# On Railway (cloud), leave it unset — the route will return a
# friendly 503 instead of hanging on an unreachable home IP.
#
#   Local demo  → set nothing (falls back to your home IP below)
#   Railway     → don't set it, or set to empty string
#   Ngrok demo  → export ESP32_CAPTURE_URL=https://xxxx.ngrok.io/capture
#
ESP32_URL: str | None = os.environ.get("ESP32_CAPTURE_URL", "http://10.161.93.232/capture") or None
if ESP32_URL:
    print(f"[INFO] ESP-32 camera URL: {ESP32_URL}")
else:
    print("[INFO] ESP32_CAPTURE_URL not set — /predict_esp32 route will be disabled.")


# ─────────────────────────────────────────────────────────────
# LOAD CLASS DICTIONARY  {int_index: "ClassName"}
# ─────────────────────────────────────────────────────────────
class_map: dict[int, str] = {}

try:
    with open(CLASS_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["class_index"])
            label = row["class"].replace("___", " – ").replace("_", " ")
            class_map[idx] = label
    print(f"[INFO] Loaded {len(class_map)} classes from {CLASS_CSV_PATH}")
except FileNotFoundError:
    print(f"[WARNING] {CLASS_CSV_PATH} not found — class names will fall back to indices.")


# ─────────────────────────────────────────────────────────────
# LOAD KERAS MODEL
# ─────────────────────────────────────────────────────────────
model: tf.keras.Model = None  # type: ignore

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[INFO] Model loaded from {MODEL_PATH}")
    print(f"[INFO] Input shape: {model.input_shape}  |  Output classes: {model.output_shape[-1]}")
except Exception as exc:
    print(f"[ERROR] Could not load model: {exc}")
    print("[INFO]  Place 'convnext_plant_disease.keras' in the same directory as app.py and restart.")


# ─────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """
    Resize → RGB → float32 array → add batch dim.
    ConvNeXt expects pixel values in [0, 255].  Do NOT normalise.
    Returns shape (1, 224, 224, 3).
    """
    img = pil_img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)          # (224, 224, 3)  range [0, 255]
    arr = np.expand_dims(arr, axis=0)               # (1, 224, 224, 3)
    return arr


# ─────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────
def _find_gradcam_layer(mdl: tf.keras.Model) -> tf.keras.layers.Layer:
    """
    Try to find the requested layer by exact name first.
    Then search inside any nested Model layer (e.g. ConvNeXt backbone).
    Finally fall back to the last Conv2D / DepthwiseConv2D-like layer,
    explicitly skipping InputLayer which also has a 4-D output shape.
    """
    # 1. Exact name match at top level
    for layer in mdl.layers:
        if layer.name == GRADCAM_LAYER:
            return layer

    # 2. Search inside nested sub-models (e.g. the ConvNeXt backbone)
    for layer in mdl.layers:
        if isinstance(layer, tf.keras.Model):
            for sub in layer.layers:
                if sub.name == GRADCAM_LAYER:
                    return sub

    # 3. Fallback: last Conv2D or DepthwiseConv2D at the top level
    target = None
    for layer in mdl.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue   # InputLayer has 4-D output – skip it
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.DepthwiseConv2D)):
            target = layer

    # 4. Broader fallback: last non-input layer with a 4-D output shape
    if target is None:
        for layer in mdl.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            try:
                shape = layer.output_shape
                if isinstance(shape, list):
                    shape = shape[0]
                if len(shape) == 4:
                    target = layer
            except Exception:
                continue

    if target is None:
        raise ValueError("No suitable Conv layer found for Grad-CAM.")
    print(f"[INFO] Grad-CAM layer fallback → '{target.name}'")
    return target


def _heatmap_to_b64(pil_img: Image.Image, heatmap: np.ndarray) -> str:
    """Shared helper: resize heatmap → JET colourmap → blend → Base64 JPEG."""
    orig_w, orig_h = pil_img.size
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    orig_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(orig_bgr, 0.60, heatmap_colored, 0.40, 0)
    _, buffer = cv2.imencode(".jpg", superimposed, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buffer).decode("utf-8")


def compute_gradcam(
    pil_img: Image.Image,
    preprocessed: np.ndarray,
    pred_class_idx: int,
) -> str:
    """
    Compute Grad-CAM heatmap, superimpose on original image,
    and return Base64-encoded JPEG string.

    Primary path: Grad-CAM via a sub-model that exposes both the last
    convolutional feature map and the final predictions.

    Fallback path (always succeeds): gradient saliency w.r.t. the input
    image, which does not require building any sub-model.
    """
    if model is None:
        raise RuntimeError("Model not loaded.")

    img_tensor = tf.cast(preprocessed, tf.float32)

    # ── Primary: proper Grad-CAM ────────────────────────────────────────
    try:
        gradcam_layer = _find_gradcam_layer(model)

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[gradcam_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            class_score = predictions[:, pred_class_idx]

        grads       = tape.gradient(class_score, conv_outputs)   # (1,H,W,C)
        pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))      # (C,)
        feature_map = conv_outputs[0]                            # (H,W,C)
        heatmap     = tf.squeeze(tf.nn.relu(
            feature_map @ pooled[..., tf.newaxis]                # (H,W,1)
        )).numpy()

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        print(f"[INFO] Grad-CAM succeeded using layer '{gradcam_layer.name}'")
        return _heatmap_to_b64(pil_img, heatmap)

    except Exception as exc:
        print(f"[WARN] Grad-CAM sub-model approach failed ({exc}); "
              "falling back to gradient saliency.")

    # ── Fallback: gradient saliency w.r.t. input ────────────────────────
    img_var = tf.Variable(img_tensor)
    with tf.GradientTape() as tape:
        preds      = model(img_var, training=False)
        class_score = preds[:, pred_class_idx]

    grads   = tape.gradient(class_score, img_var)          # (1,224,224,3)
    heatmap = tf.reduce_mean(tf.abs(grads[0]), axis=-1)    # (224,224)
    heatmap = tf.nn.relu(heatmap).numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return _heatmap_to_b64(pil_img, heatmap)


# ─────────────────────────────────────────────────────────────
# SHARED INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────
def run_inference(pil_img: Image.Image) -> dict:
    """
    Full pipeline: preprocess → predict → Grad-CAM → return result dict.
    Grad-CAM failure is non-fatal; inference result is always returned.
    """
    if model is None:
        raise RuntimeError("Model is not loaded. Check app startup logs.")

    preprocessed = preprocess_image(pil_img)
    predictions  = model.predict(preprocessed, verbose=0)[0]   # (num_classes,)

    pred_idx    = int(np.argmax(predictions))
    confidence  = float(predictions[pred_idx]) * 100.0
    disease_raw = class_map.get(pred_idx, f"Class_{pred_idx}")

    # Grad-CAM visualisation — non-fatal: a heatmap error must not kill the prediction
    gradcam_b64 = ""
    try:
        gradcam_b64 = compute_gradcam(pil_img, preprocessed, pred_idx)
    except Exception as exc:
        print(f"[WARN] Grad-CAM completely failed, returning empty heatmap: {exc}")

    return {
        "disease":        disease_raw,
        "confidence":     round(confidence, 2),
        "gradcam_base64": gradcam_b64,
    }


# ─────────────────────────────────────────────────────────────
# ROUTE 1 — /predict_upload
# ─────────────────────────────────────────────────────────────
@app.post("/predict_upload")
async def predict_upload(file: UploadFile = File(...)):
    """
    Accept an uploaded image file and return plant disease prediction
    along with a Grad-CAM explanation heatmap.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        raw_bytes = await file.read()
        pil_img   = Image.open(io.BytesIO(raw_bytes))
        result    = run_inference(pil_img)
        return JSONResponse(content=result)

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


# ─────────────────────────────────────────────────────────────
# ROUTE 2 — /predict_esp32
# ─────────────────────────────────────────────────────────────
@app.post("/predict_esp32")
async def predict_esp32():
    """
    Fetch a JPEG snapshot from the ESP-32 IP camera, run inference + Grad-CAM,
    and return results.

    Requires the ESP32_CAPTURE_URL environment variable to be set.
    If not set (e.g. on Railway cloud), returns HTTP 503 immediately
    instead of hanging on an unreachable home-network IP.
    """
    # ── Guard: route disabled when no ESP32 URL is configured ──
    if not ESP32_URL:
        raise HTTPException(
            status_code=503,
            detail=(
                "ESP-32 camera is not configured on this deployment. "
                "Set the ESP32_CAPTURE_URL environment variable to enable this route."
            ),
        )

    try:
        resp = requests.get(ESP32_URL, timeout=5)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="ESP-32 camera timed out. Check that the device is online and on the same network.",
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot reach ESP-32 camera. Verify the IP address and network connection.",
        )
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"ESP-32 request error: {exc}")

    try:
        # Decode JPEG bytes via OpenCV
        img_array = np.frombuffer(resp.content, dtype=np.uint8)
        img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from ESP-32 response.")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        result  = run_inference(pil_img)
        return JSONResponse(content=result)

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


# ─────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "num_classes":  len(class_map),
    }


# ─────────────────────────────────────────────────────────────
# SERVE FRONTEND  (must be LAST — after all API routes)
# ─────────────────────────────────────────────────────────────
# Root → index.html
@app.get("/", include_in_schema=False)
async def serve_root():
    return FileResponse("index.html")

# dashboard.html → explicit route so Railway can navigate there
@app.get("/dashboard.html", include_in_schema=False)
async def serve_dashboard():
    return FileResponse("dashboard.html")

# All other static assets (app.js, style.css, hero_bg.png, …)
# IMPORTANT: mount at "/static" internally, but the HTML already
# references files at the root path, so we mount at "/" last.
# FastAPI checks named routes first, so API endpoints are safe.
app.mount("/", StaticFiles(directory=".", html=True), name="frontend")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT (for direct python app.py execution)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
