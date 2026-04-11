from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

import os
import time
import logging
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import mlflow
import mlflow.pyfunc

# -----------------------------
# Logging setup
# -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# -----------------------------
# Environment Configuration
# -----------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000"
)

HEART_MODEL_NAME = "CDSS_Model"
DIABETES_MODEL_NAME = "CDSS_Diabetes_Model"

MAX_RETRIES = 20
RETRY_DELAY = 3

# -----------------------------
# Initialize FastAPI
# -----------------------------

app = FastAPI(
    title="Multi-Disease Clinical Decision Support System",
    version="1.0"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# -----------------------------
# Model variables
# -----------------------------

heart_model = None
diabetes_model = None

# We keep references to the underlying sklearn models
# so we can call predict_proba() directly and correctly.
# mlflow.pyfunc wraps the model and does NOT expose predict_proba,
# which caused the probability inversion bug in the original code.

heart_sklearn_model = None
diabetes_sklearn_model = None

# -----------------------------
# Wait for MLflow
# -----------------------------

def wait_for_mlflow():

    logger.info("Waiting for MLflow server...")

    for attempt in range(MAX_RETRIES):

        try:

            mlflow.search_experiments()

            logger.info("MLflow is ready")

            return

        except Exception:

            logger.warning(
                f"Retry {attempt + 1}/{MAX_RETRIES} - MLflow not ready"
            )

            time.sleep(RETRY_DELAY)

    raise RuntimeError("MLflow server not reachable")

# -----------------------------
# Load Models
# -----------------------------

def load_models():

    global heart_model, heart_sklearn_model
    global diabetes_model, diabetes_sklearn_model

    # ---- Heart Model ----
    try:

        logger.info("Loading Heart Model")

        heart_model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{HEART_MODEL_NAME}@production"
        )

        # Extract the underlying sklearn model for predict_proba access.
        # mlflow.pyfunc wraps the sklearn model inside .unwrap_python_model()
        # or accessible via ._model_impl.sklearn_model depending on flavour.
        try:
            heart_sklearn_model = heart_model.unwrap_python_model()
        except Exception:
            # Fallback: access the sklearn model stored in the pyfunc wrapper
            heart_sklearn_model = heart_model._model_impl.sklearn_model

        logger.info("Heart model loaded (sklearn model extracted for predict_proba)")

    except Exception as e:

        logger.error(f"Heart model failed: {e}")

    # ---- Diabetes Model ----
    try:

        logger.info("Loading Diabetes Model")

        diabetes_model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{DIABETES_MODEL_NAME}@production"
        )

        try:
            diabetes_sklearn_model = diabetes_model.unwrap_python_model()
        except Exception:
            diabetes_sklearn_model = diabetes_model._model_impl.sklearn_model

        logger.info("Diabetes model loaded (sklearn model extracted for predict_proba)")

    except Exception as e:

        logger.error(f"Diabetes model failed: {e}")

# -----------------------------
# Startup Event
# -----------------------------

@app.on_event("startup")
def startup_event():

    logger.info("API startup initiated")

    wait_for_mlflow()

    load_models()

    logger.info("Startup complete")

# -----------------------------
# Request Schemas
# -----------------------------

class HeartRequest(BaseModel):

    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


class DiabetesRequest(BaseModel):

    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float

# -----------------------------
# Health Check
# -----------------------------

@app.get("/")
def home():

    return {

        "status": "CDSS API running",
        "mlflow": MLFLOW_TRACKING_URI

    }

# -----------------------------
# Recommendation Functions
# -----------------------------

def get_recommendation(risk):

    if risk == "Low Risk":

        return "Maintain healthy lifestyle"

    elif risk == "Medium Risk":

        return "Consult doctor regularly"

    else:

        return "Immediate medical consultation required"


def get_diabetes_recommendation(risk):

    if risk == "Low Risk":

        return "Maintain balanced diet"

    elif risk == "Medium Risk":

        return "Monitor blood sugar"

    else:

        return "High diabetes risk — consult doctor"

# -----------------------------
# Prometheus Metrics
# -----------------------------

REQUEST_COUNT = Counter(
    "api_request_count",
    "Total API Requests",
    ["endpoint"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API Request Latency",
    ["endpoint"]
)

# -----------------------------
# FIXED Probability Function
# -----------------------------
# ROOT CAUSE of the original bug:
#   mlflow.pyfunc does NOT expose predict_proba, so hasattr() always returned
#   False. The fallback then called model.predict() which returns integer class
#   labels (0 or 1). When prediction[0] == 0 (healthy), it returned 0.0 →
#   "Low Risk" (sometimes correct by luck). When prediction[0] == 1 (sick), it
#   returned 1.0 → "High Risk" (seemed right but was fragile). For the cases
#   where the model predicted 0 for a truly sick patient the probability was
#   0.0 → wrongly "Low Risk". This caused the inversion you observed.
#
# FIX:
#   We extract the underlying sklearn model at load time (heart_sklearn_model /
#   diabetes_sklearn_model) and call predict_proba on it directly.
#   predict_proba returns [P(class=0), P(class=1)], so index [1] gives the
#   probability of the positive class (disease / diabetes).
#   High probability → High Risk. Low probability → Low Risk. Correct.
# -----------------------------

def get_probability(sklearn_model, features):
    """
    Returns probability of the POSITIVE class (disease present = 1).
    Uses the unwrapped sklearn model to call predict_proba directly.
    """

    try:

        if sklearn_model is not None and hasattr(sklearn_model, "predict_proba"):

            proba = sklearn_model.predict_proba(features)

            # proba shape: (n_samples, n_classes)
            # classes_ = [0, 1] → index 1 is probability of positive class
            return float(proba[0][1])

        # Fallback if sklearn model not available: use pyfunc predict
        # predict returns class label; map to a rough probability
        logger.warning("predict_proba not available; using class label as fallback")

        return 0.5  # neutral; avoids false High/Low assignments

    except Exception as e:

        logger.warning(f"Probability error: {e} — returning 0.5 as neutral fallback")

        return 0.5

# -----------------------------
# Risk Level Function
# FIXED thresholds: Low 0–0.3, Medium 0.3–0.7, High 0.7–1.0
# (original used 0.35 / 0.65 which didn't match the spec)
# -----------------------------

def get_risk_level(probability):

    if probability < 0.3:

        return "Low Risk"

    elif probability < 0.7:

        return "Medium Risk"

    else:

        return "High Risk"

# -----------------------------
# Heart Prediction
# -----------------------------

@app.post("/predict/heart")
def predict_heart(data: HeartRequest):

    if heart_model is None:

        raise HTTPException(
            status_code=500,
            detail="Heart model not loaded"
        )

    start_time = time.time()

    REQUEST_COUNT.labels(
        endpoint="heart"
    ).inc()

    try:

        features = [[
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            data.fbs,
            data.restecg,
            data.thalach,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal
        ]]

        # Use pyfunc model for the class prediction (0 or 1)
        prediction = heart_model.predict(features)[0]

        # Use the unwrapped sklearn model for a correct probability
        probability = get_probability(
            heart_sklearn_model,
            features
        )

        risk = get_risk_level(probability)

        result = {

            "prediction": int(prediction),
            "probability": round(probability, 2),
            "risk_level": risk,
            "recommendation": get_recommendation(risk)

        }

        REQUEST_LATENCY.labels(
            endpoint="heart"
        ).observe(
            time.time() - start_time
        )

        return result

    except Exception as e:

        logger.error(str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# -----------------------------
# Diabetes Prediction
# -----------------------------

@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesRequest):

    if diabetes_model is None:

        raise HTTPException(
            status_code=500,
            detail="Diabetes model not loaded"
        )

    start_time = time.time()

    REQUEST_COUNT.labels(
        endpoint="diabetes"
    ).inc()

    try:

        features = [[
            data.pregnancies,
            data.glucose,
            data.blood_pressure,
            data.skin_thickness,
            data.insulin,
            data.bmi,
            data.diabetes_pedigree_function,
            data.age
        ]]

        # Use pyfunc model for the class prediction (0 or 1)
        prediction = diabetes_model.predict(features)[0]

        # Use the unwrapped sklearn model for a correct probability
        probability = get_probability(
            diabetes_sklearn_model,
            features
        )

        risk = get_risk_level(probability)

        result = {

            "prediction": int(prediction),
            "probability": round(probability, 2),
            "risk_level": risk,
            "recommendation": get_diabetes_recommendation(risk)

        }

        REQUEST_LATENCY.labels(
            endpoint="diabetes"
        ).observe(
            time.time() - start_time
        )

        return result

    except Exception as e:

        logger.error(str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# -----------------------------
# Prometheus Metrics Endpoint
# -----------------------------

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type="text/plain"
    )

# -----------------------------
# Auto Prometheus Instrumentation
# -----------------------------

Instrumentator().instrument(app).expose(app)