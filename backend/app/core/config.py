from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "artifacts" / "best_model.keras"
CLASSES_PATH = ROOT / "artifacts" / "class_names.json"

IMG_SIZE = 128
MAX_UPLOAD_MB = 10
ALLOWED_TYPES = {"image/jpeg", "image/png"}
CORS_ORIGINS = ["http://localhost:4200"]

CLASS_NAMES = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}