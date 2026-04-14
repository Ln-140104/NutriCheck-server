import os, uuid, shutil, tempfile
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Loading YOLO model...")
    # Import here to avoid any startup-time issues
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print(f"Model loaded: {model.names}")
    yield

app = FastAPI(title="NutriCheck API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

def safe_predict(image_path, conf=0.15):
    """
    Run prediction and return a flat list of (class_name, mask_numpy) tuples.
    This function handles ALL ultralytics API versions safely.
    """
    results = model.predict(image_path, imgsz=640, conf=conf, verbose=False)
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    detections = []

    for result in results:
        # No masks at all — skip
        if result.masks is None:
            continue

        num_detections = len(result.boxes.cls)

        for i in range(num_detections):
            cls_id   = int(result.boxes.cls[i])
            cls_name = result.names[cls_id].lower()

            # ── Safely extract mask tensor ────────────────────────────────
            try:
                # This works on all ultralytics versions
                mask_tensor = result.masks.data[i]
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
            except Exception:
                # Fallback: try .masks[i].data
                try:
                    mask_np = result.masks[i].data.cpu().numpy().astype(np.uint8) * 255
                except Exception:
                    continue  # skip this detection if mask extraction fails

            # Resize mask to match original image dimensions
            if mask_np.ndim == 3:
                mask_np = mask_np[0]  # remove channel dim if present
            mask_np = cv2.resize(
                mask_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST
            )
            detections.append((cls_name, mask_np))

    return detections, img_h, img_w


def detect_scale(image_path):
    detections, img_h, img_w = safe_predict(image_path, conf=0.3)
    coin_candidates = []

    for cls_name, mask_np in detections:
        if "coin" not in cls_name:
            continue
        contours, _ = cv2.findContours(
            mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            if cv2.contourArea(c) > 100:
                coin_candidates.append(c)

    if not coin_candidates:
        raise ValueError(
            "No coin detected. Make sure the Rs.10 coin is clearly "
            "visible in the top-view photo."
        )

    coin_candidates.sort(key=cv2.contourArea, reverse=True)
    (_, _), radius = cv2.minEnclosingCircle(coin_candidates[0])
    if radius < 1:
        raise ValueError("Coin too small in image. Move camera closer.")
    return COIN_DIAMETER_CM / (2 * radius)


def get_food_mask(image_path, food_key):
    food = FOOD_DB[food_key]
    detections, img_h, img_w = safe_predict(image_path, conf=0.15)
    mask_total = np.zeros((img_h, img_w), dtype=np.uint8)

    for cls_name, mask_np in detections:
        if food["class_name"] in cls_name:
            mask_total = cv2.bitwise_or(mask_total, mask_np)

    if np.sum(mask_total) == 0:
        raise ValueError(
            f"No {food['name']} detected. "
            "Try better lighting, clearer background, or move closer."
        )
    return mask_total


def measure_top(mask_total, scale):
    ys, xs = np.where(mask_total > 0)
    points  = np.column_stack([xs, ys]).astype(np.float32)
    _, eigenvectors = cv2.PCACompute(points, mean=None)
    proj_long  = points @ eigenvectors[0]
    proj_short = points @ eigenvectors[1]
    L = float(proj_long.max()  - proj_long.min())  * scale
    W = float(proj_short.max() - proj_short.min()) * scale
    return L, W


def measure_side(mask_total, scale):
    ys, _ = np.where(mask_total > 0)
    return float(np.max(ys) - np.min(ys)) * scale


def compute_volume(food_key, L, W, H):
    shape = FOOD_DB[food_key]["shape"]
    if shape == "ellipsoid": return (np.pi / 6) * L * W * H
    if shape == "cuboid":    return L * W * H
    raise ValueError(f"Unknown shape: {shape}")


def compute_nutrients(food_key, weight_g):
    return {
        k: round(v * weight_g / 100.0, 2)
        for k, v in FOOD_DB[food_key]["nutrition"].items()
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "classes": model.names if model else "still loading"
    }

@app.get("/foods")
def list_foods():
    return {
        k: {"name": v["name"], "nutrition_per_100g": v["nutrition"]}
        for k, v in FOOD_DB.items()
    }

@app.post("/estimate")
async def estimate(
    top_view:  UploadFile = File(...),
    side_view: UploadFile = File(...),
    food:      str        = Form(...),
):
    if model is None:
        raise HTTPException(503, "Model still loading. Wait 30s and try again.")
    if food not in FOOD_DB:
        raise HTTPException(400, f"Unknown food '{food}'. Use: {list(FOOD_DB.keys())}")

    tmpdir = tempfile.mkdtemp()
    try:
        top_path  = os.path.join(tmpdir, f"top_{uuid.uuid4().hex}.jpg")
        side_path = os.path.join(tmpdir, f"side_{uuid.uuid4().hex}.jpg")

        contents = await top_view.read()
        with open(top_path, "wb") as f:
            f.write(contents)

        contents = await side_view.read()
        with open(side_path, "wb") as f:
            f.write(contents)

        scale     = detect_scale(top_path)
        top_mask  = get_food_mask(top_path,  food)
        side_mask = get_food_mask(side_path, food)
        L, W      = measure_top(top_mask,  scale)
        H         = measure_side(side_mask, scale)
        volume    = compute_volume(food, L, W, H)
        weight    = volume * FOOD_DB[food]["density"]
        nutrients = compute_nutrients(food, weight)

        return JSONResponse({
            "success": True,
            "data": {
                "food":            FOOD_DB[food]["name"],
                "dimensions_cm":   {
                    "length": round(L, 3),
                    "width":  round(W, 3),
                    "height": round(H, 3),
                },
                "volume_cm3":      round(volume, 4),
                "weight_g":        round(weight, 2),
                "nutrients":       nutrients,
                "scale_cm_per_px": round(scale, 6),
            }
        })

    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        # Return full error so we can debug
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR: {tb}")
        raise HTTPException(500, f"Server error: {str(e)}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
