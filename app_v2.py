import os, uuid, shutil, tempfile
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from contextlib import asynccontextmanager

FOOD_DB = {
    "almond": {
        "name": "Almond", "class_name": "almond",
        "density": 0.64, "shape": "ellipsoid",
        "nutrition": {"protein": 21.15, "fat": 49.42, "carbs": 21.55, "calories": 579.0},
    },
    "paneer": {
        "name": "Paneer", "class_name": "paneer",
        "density": 1.10, "shape": "cuboid",
        "nutrition": {"protein": 18.3, "fat": 20.8, "carbs": 1.2, "calories": 265.0},
    },
}

COIN_DIAMETER_CM = 2.7
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
model = None   # loaded after startup

# ── Load model AFTER server binds to port ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("⏳ Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded — classes: {model.names}")
    yield
    print("Server shutting down.")

app = FastAPI(title="NutriCheck API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Helper functions ──────────────────────────────────────────────────────────

def detect_scale(image_path):
    results = model.predict(image_path, imgsz=640, conf=0.3, verbose=False)
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    coin_candidates = []
    for result in results:
        if result.masks is None:
            continue
        for i in range(len(result.boxes.cls)):
            cls_name = result.names[int(result.boxes.cls[i])].lower()
            if "coin" in cls_name:
                mask_np = result.masks.data[i].cpu().numpy().astype(np.uint8) * 255
                mask_np = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) > 100:
                        coin_candidates.append(c)
    if not coin_candidates:
        raise ValueError("No coin detected. Make sure the Rs.10 coin is clearly visible.")
    coin_candidates.sort(key=cv2.contourArea, reverse=True)
    (_, _), radius = cv2.minEnclosingCircle(coin_candidates[0])
    return COIN_DIAMETER_CM / (2 * radius)


def get_food_mask(image_path, food_key):
    food = FOOD_DB[food_key]
    results = model.predict(image_path, imgsz=640, conf=0.15, verbose=False)
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    mask_total = np.zeros((img_h, img_w), dtype=np.uint8)
    for result in results:
        if result.masks is None:
            continue
        for i in range(len(result.boxes.cls)):
            cls_name = result.names[int(result.boxes.cls[i])].lower()
            if food["class_name"] in cls_name:
                mask_np = result.masks.data[i].cpu().numpy().astype(np.uint8) * 255
                mask_np = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                mask_total = cv2.bitwise_or(mask_total, mask_np)
    if np.sum(mask_total) == 0:
        raise ValueError(f"No {food['name']} detected. Try better lighting or a clearer photo.")
    return mask_total


def measure_top(mask_total, food_key, scale):
    ys, xs = np.where(mask_total > 0)
    points = np.column_stack([xs, ys]).astype(np.float32)
    _, eigenvectors = cv2.PCACompute(points, mean=None)
    proj_long  = points @ eigenvectors[0]
    proj_short = points @ eigenvectors[1]
    return float(proj_long.max() - proj_long.min()) * scale, \
           float(proj_short.max() - proj_short.min()) * scale


def measure_side(mask_total, scale):
    ys, _ = np.where(mask_total > 0)
    return float(np.max(ys) - np.min(ys)) * scale


def compute_volume(food_key, L, W, H):
    shape = FOOD_DB[food_key]["shape"]
    if shape == "ellipsoid": return (np.pi / 6) * L * W * H
    if shape == "cuboid":    return L * W * H
    raise ValueError(f"Unknown shape: {shape}")


def compute_nutrients(food_key, weight_g):
    return {k: round(v * weight_g / 100.0, 2)
            for k, v in FOOD_DB[food_key]["nutrition"].items()}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "classes": model.names if model else "loading"}


@app.get("/foods")
def list_foods():
    return {k: {"name": v["name"], "nutrition_per_100g": v["nutrition"]}
            for k, v in FOOD_DB.items()}


@app.post("/estimate")
async def estimate(
    top_view:  UploadFile = File(...),
    side_view: UploadFile = File(...),
    food:      str        = Form(...),
):
    if model is None:
        raise HTTPException(503, "Model is still loading, please try again in 30 seconds.")
    if food not in FOOD_DB:
        raise HTTPException(400, f"Unknown food '{food}'")

    tmpdir = tempfile.mkdtemp()
    try:
        top_path  = os.path.join(tmpdir, f"top_{uuid.uuid4().hex}.jpg")
        side_path = os.path.join(tmpdir, f"side_{uuid.uuid4().hex}.jpg")
        with open(top_path,  "wb") as f: f.write(await top_view.read())
        with open(side_path, "wb") as f: f.write(await side_view.read())

        scale     = detect_scale(top_path)
        top_mask  = get_food_mask(top_path,  food)
        side_mask = get_food_mask(side_path, food)
        L, W      = measure_top(top_mask,  food, scale)
        H         = measure_side(side_mask, scale)
        volume    = compute_volume(food, L, W, H)
        weight    = volume * FOOD_DB[food]["density"]
        nutrients = compute_nutrients(food, weight)

        return JSONResponse({"success": True, "data": {
            "food":            FOOD_DB[food]["name"],
            "dimensions_cm":   {"length": round(L,3), "width": round(W,3), "height": round(H,3)},
            "volume_cm3":      round(volume, 4),
            "weight_g":        round(weight, 2),
            "nutrients":       nutrients,
            "scale_cm_per_px": round(scale, 6),
        }})
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Server error: {str(e)}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
