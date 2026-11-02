"""FastAPI service: upload a car photo, get the predicted brand and a confidence score.

Two stages:
    1. "Car gate": an ImageNet-pretrained ResNet18 estimates how likely the image shows a car.
       If it does not look like a car (cat, food, person ...), we stop and say so.
    2. Brand classifier: our own trained ResNet18 predicts one of the 20 brands.

Run from the project root:
    uvicorn api.main:app

Then open http://127.0.0.1:8000/docs to try it in the browser.
"""

import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import build_model  # noqa: E402  (needs the path line above)

CHECKPOINT = ROOT / "models" / "resnet18_best.pth"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 8 * 1024 * 1024      # Reject files bigger than 8 MB.
LOW_CONFIDENCE = 0.50            # Below this, the brand answer is probably unreliable.

# Car gate settings. CAR_THRESHOLD is chosen with tests/check_car_gate.py.
CAR_THRESHOLD = 0.20
# ImageNet class names that mean "a car or a part of a car".
CAR_CLASSES = {
    "ambulance", "beach wagon", "cab", "convertible", "jeep", "limousine", "minivan",
    "Model T", "pickup", "racer", "sports car", "police van", "minibus", "moving van",
    "tow truck", "car wheel", "grille",
}

state: dict = {}                 # Holds the models, class names and transforms.


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models ONCE when the server starts (not on every request)."""
    # Stage 2: our brand classifier.
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], n_classes=len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()  # Evaluation mode: no training behaviour.
    state["model"] = model
    state["class_names"] = ckpt["class_names"]
    # Exactly the same preprocessing that was used for validation/test images.
    state["transform"] = transforms.Compose([
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["mean"], ckpt["std"]),
    ])

    # Stage 1: the car gate (ImageNet ResNet18, used only to answer "is this a car?").
    gate_weights = models.ResNet18_Weights.DEFAULT
    state["gate"] = models.resnet18(weights=gate_weights).eval()
    state["gate_transform"] = gate_weights.transforms()
    names = gate_weights.meta["categories"]
    state["gate_ids"] = [i for i, n in enumerate(names) if n in CAR_CLASSES]
    yield
    state.clear()


app = FastAPI(title="Car Brand Classifier", version="1.1", lifespan=lifespan)

# Allow the React development server to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    """Send visitors of the bare address to the interactive docs page."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    """Quick check that the server and models are ready."""
    return {"status": "ok", "classes": len(state.get("class_names", []))}


def car_probability(image: Image.Image) -> float:
    """How likely is it that the image shows a car? (sum of car-related ImageNet classes)"""
    batch = state["gate_transform"](image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(state["gate"](batch), dim=1)[0]
    return float(probs[state["gate_ids"]].sum())


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Take an image, return the predicted car brand and the model's confidence."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Please upload a JPEG, PNG or WebP image.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is too large (maximum 8 MB).")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "The file is not a valid image.")

    # Stage 1: is it a car at all?
    car_prob = car_probability(image)
    if car_prob < CAR_THRESHOLD:
        return {
            "is_car": False,
            "car_probability": round(car_prob, 4),
            "predicted_class": None,
            "confidence": None,
            "top3": [],
            "message": "This image does not look like a car, so no brand was predicted.",
        }

    # Stage 2: which brand?
    batch = state["transform"](image).unsqueeze(0)           # Shape: 1 x 3 x 224 x 224
    with torch.no_grad():
        probs = torch.softmax(state["model"](batch), dim=1)[0]  # Scores -> probabilities.
    top = torch.topk(probs, k=3)
    names = state["class_names"]
    top3 = [{"class": names[i], "confidence": round(float(p), 4)}
            for p, i in zip(top.values, top.indices)]

    result = {
        "is_car": True,
        "car_probability": round(car_prob, 4),
        "predicted_class": top3[0]["class"],
        "confidence": top3[0]["confidence"],
        "top3": top3,
    }
    if result["confidence"] < LOW_CONFIDENCE:
        result["warning"] = ("Low confidence: this may not be a car from the 20 supported "
                             "brands, or the photo is unclear.")
    return result