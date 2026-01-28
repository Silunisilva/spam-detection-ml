import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import joblib

MODEL_PATH = "model.joblib"
VECT_PATH = "vectorizer.joblib"

app = FastAPI(title="Spam Classifier API")

model = None
vectorizer = None


def load_artifacts():
    global model, vectorizer
    if model is None or vectorizer is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
            raise RuntimeError("Model artifacts not found. Run train.py first to produce model.joblib and vectorizer.joblib")
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECT_PATH)


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        load_artifacts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    vec = vectorizer.transform([req.text])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0].tolist() if hasattr(model, "predict_proba") else None
    return {"label": int(pred), "label_str": "spam" if pred == 1 else "ham", "probabilities": probs}


# mount static files AFTER API routes (serves static/index.html)
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
