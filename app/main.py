from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

import os
import time
import logging
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# -----------------------------
# Model variables
# -----------------------------

heart_model = None
diabetes_model = None

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

        try:
            heart_sklearn_model = heart_model.unwrap_python_model()
        except Exception:
            heart_sklearn_model = heart_model._model_impl.sklearn_model

        logger.info(f"Heart sklearn model type: {type(heart_sklearn_model)}")
        logger.info(f"Heart classes_: {heart_sklearn_model.classes_}")
        logger.info("Heart model loaded successfully")

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

        logger.info(f"Diabetes sklearn model type: {type(diabetes_sklearn_model)}")
        logger.info(f"Diabetes classes_: {diabetes_sklearn_model.classes_}")
        logger.info("Diabetes model loaded successfully")

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
# Heart Probability Function
# -----------------------------
# KEY FINDING: In heart.csv, target labels are:
#   0 = heart disease PRESENT
#   1 = no heart disease
# This is INVERTED from standard convention.
# Therefore:
#   proba[0][0] = probability of disease (what we want for risk)
#   proba[0][1] = probability of no disease (NOT what we want)
# So we use proba[0][0] as the heart disease risk probability.
# -----------------------------

def get_heart_probability(sklearn_model, features):

    try:

        if sklearn_model is not None and hasattr(sklearn_model, "predict_proba"):

            proba = sklearn_model.predict_proba(features)

            # class 0 = disease present → use index 0
            disease_probability = float(proba[0][0])

            logger.info(
                f"Heart raw proba: {proba} → P(disease)={disease_probability}"
            )

            return disease_probability

        logger.warning("Heart predict_proba not available — returning 0.5")

        return 0.5

    except Exception as e:

        logger.warning(f"Heart probability error: {e} — returning 0.5")

        return 0.5

# -----------------------------
# Diabetes Probability Function
# -----------------------------
# In diabetes.csv, Outcome labels are STANDARD:
#   0 = no diabetes
#   1 = diabetic
# Therefore:
#   proba[0][1] = probability of diabetes (what we want for risk)
#   proba[0][0] = probability of no diabetes (NOT what we want)
# So we use proba[0][1] as the diabetes risk probability.
# -----------------------------

def get_diabetes_probability(sklearn_model, features):

    try:

        if sklearn_model is not None and hasattr(sklearn_model, "predict_proba"):

            proba = sklearn_model.predict_proba(features)

            # class 1 = diabetic → use index 1
            disease_probability = float(proba[0][1])

            logger.info(
                f"Diabetes raw proba: {proba} → P(diabetic)={disease_probability}"
            )

            return disease_probability

        logger.warning("Diabetes predict_proba not available — returning 0.5")

        return 0.5

    except Exception as e:

        logger.warning(f"Diabetes probability error: {e} — returning 0.5")

        return 0.5

# -----------------------------
# Risk Level Function
# Low    : 0.0 – 0.3
# Medium : 0.3 – 0.7
# High   : 0.7 – 1.0
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

        features = pd.DataFrame([{
            "age": data.age,
            "sex": data.sex,
            "cp": data.cp,
            "trestbps": data.trestbps,
            "chol": data.chol,
            "fbs": data.fbs,
            "restecg": data.restecg,
            "thalach": data.thalach,
            "exang": data.exang,
            "oldpeak": data.oldpeak,
            "slope": data.slope,
            "ca": data.ca,
            "thal": data.thal
        }])

        # pyfunc model for class prediction
        prediction = heart_model.predict(features)[0]

        # Heart-specific: class 0 = disease → proba[0][0]
        probability = get_heart_probability(
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

        features = pd.DataFrame([{
            "Pregnancies": data.pregnancies,
            "Glucose": data.glucose,
            "BloodPressure": data.blood_pressure,
            "SkinThickness": data.skin_thickness,
            "Insulin": data.insulin,
            "BMI": data.bmi,
            "DiabetesPedigreeFunction": data.diabetes_pedigree_function,
            "Age": data.age
        }])

        # pyfunc model for class prediction
        prediction = diabetes_model.predict(features)[0]

        # Diabetes-specific: class 1 = diabetic → proba[0][1]
        probability = get_diabetes_probability(
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