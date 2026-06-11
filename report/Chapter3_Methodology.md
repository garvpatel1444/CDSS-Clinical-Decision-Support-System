# CHAPTER 3
# METHODOLOGY

This chapter describes the complete methodology followed in the design, development, and deployment of the Clinical Decision Support System. It covers the system architecture, data collection and preprocessing, model training and evaluation, API deployment, containerisation, and real-time monitoring. The methodology follows a structured machine learning engineering approach to ensure accuracy, reproducibility, and operational reliability.

---

## 3.1 System Architecture

The CDSS is built as a multi-tier, containerised web application. The system is composed of seven key components that work together to deliver real-time disease risk prediction.

The **Frontend** provides a web-based interface where users submit patient health parameters and receive a risk classification result. It communicates with the backend via HTTP POST requests carrying patient data in JSON format.

The **FastAPI Backend** acts as the core application server. It hosts dedicated prediction endpoints for heart disease and diabetes, performs input validation, applies preprocessing transformations, invokes the trained models, and returns structured risk classification responses.

The **Machine Learning Models** — two independently trained Random Forest classifiers — are loaded at application startup and serve predictions at runtime. The **MLflow Model Registry** manages all model versions, logging training metrics, hyperparameters, and serialised model artefacts in a centralised and auditable store.

The **Docker Deployment** layer containerises each service individually. Docker Compose orchestrates the complete multi-service stack, configuring service networking, port mappings, and volume mounts within a single declarative configuration file. The **Prometheus** service scrapes the API metrics endpoint at fifteen-second intervals, and **Grafana** visualises the collected metrics on a real-time operational dashboard.

[INSERT FIGURE 3.1: System Architecture Diagram]

---

## 3.2 Data Collection

Two publicly available, clinically validated datasets are used to train the respective prediction models.

The **Cleveland Heart Disease Dataset**, sourced from the UCI Machine Learning Repository, contains 303 patient records with 13 clinical features including age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG results, maximum heart rate, exercise-induced angina, ST depression (oldpeak), ST slope, number of major vessels, and thalassemia type. The binary target variable indicates the presence or absence of heart disease.

The **Pima Indians Diabetes Dataset**, maintained by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) and available on Kaggle, contains 768 patient records with 8 features: number of pregnancies, plasma glucose, diastolic blood pressure, triceps skin thickness, serum insulin, BMI, diabetes pedigree function, and age. The target variable indicates whether the patient is diabetic.

**Table 3.1: Dataset Summary**

| Attribute | Heart Disease Dataset | Diabetes Dataset |
|---|---|---|
| Source | UCI ML Repository | NIDDK / Kaggle |
| Total Records | 303 | 768 |
| Number of Features | 13 | 8 |
| Positive Cases | 165 (54.5%) | 268 (34.9%) |
| Negative Cases | 138 (45.5%) | 500 (65.1%) |
| Target Variable | Heart disease (0/1) | Diabetic (0/1) |

---

## 3.3 Data Preprocessing

Raw medical datasets contain quality issues that must be resolved before model training. A consistent preprocessing pipeline is applied to both datasets.

**Handling Missing Values:** The Diabetes dataset contains biologically implausible zero values in features such as Glucose, BloodPressure, SkinThickness, Insulin, and BMI, which represent missing recordings. These are replaced using column-wise median imputation, as the affected features exhibit skewed distributions where mean imputation would introduce systematic bias.

**Data Cleaning:** Duplicate records are removed from both datasets. Feature data types are explicitly cast to ensure consistency during array operations, and categorical integer-encoded features are verified for range validity.

**Feature Selection:** All available features in both datasets are retained for training based on their established clinical relevance. Post-training feature importance scores from the Random Forest model are used to validate each feature's predictive contribution.

**Data Normalisation:** `StandardScaler` is applied to standardise all features to zero mean and unit variance, preventing high-magnitude features from disproportionately influencing the model. The scaler is fitted exclusively on the training set and applied to both the training and test sets to prevent data leakage. An 80:20 stratified train-test split is used across both datasets to preserve class distribution in each partition.

---

## 3.4 Model Training

The **Random Forest Classifier** is selected as the predictive algorithm for both disease models. It is a supervised ensemble learning method that builds multiple decision trees on bootstrapped training samples, each using a random subset of features at every split node. Final predictions are determined by majority voting across all trees, and class probabilities are derived by averaging individual tree probability estimates. This ensemble mechanism reduces prediction variance and produces a robust, generalisable classifier that is well-suited to tabular clinical data.

**Train-Test Split:** Both datasets are divided into an 80% training partition and a 20% test partition using stratified sampling to maintain proportional class representation, particularly important for the class-imbalanced Diabetes dataset.

**Training Process:** Hyperparameter optimisation is conducted using `GridSearchCV` with five-fold stratified cross-validation, tuning `n_estimators`, `max_depth`, and `min_samples_split` with the weighted F1-score as the optimisation criterion. The `class_weight='balanced'` setting is applied to both models to mitigate class imbalance.

**Model Saving:** Following training, the fitted `RandomForestClassifier` and its associated `StandardScaler` are serialised to disk as `.pkl` binary files using Python's `joblib` library, and loaded directly by the FastAPI application at runtime.

**Model Versioning with MLflow:** Each training run is tracked within an MLflow experiment. Hyperparameters are logged using `mlflow.log_params()`, evaluation metrics using `mlflow.log_metrics()`, and model artefacts using `mlflow.sklearn.log_model()`. The best-performing model is registered in the MLflow Model Registry and promoted to the production stage, providing a fully auditable and reproducible model lifecycle.

---

## 3.5 Model Evaluation

Model performance is assessed on the held-out 20% test partition using standard classification evaluation metrics.

**Accuracy** measures the overall proportion of correct predictions. **Precision** quantifies the proportion of positive predictions that are genuinely positive, minimising unnecessary false alarms. **Recall (Sensitivity)** measures the proportion of actual positive cases correctly identified — this is the primary clinical metric, as a missed at-risk patient (false negative) carries the most serious clinical consequence. **F1-Score**, the harmonic mean of precision and recall, provides a balanced measure used as the primary tuning criterion. **AUC-ROC** quantifies the model's discriminative ability across all classification thresholds, independent of any specific decision boundary.

**Table 3.3: Evaluation Metric Formulas**

| Metric | Formula | Description |
|---|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Overall proportion of correct predictions |
| Precision | TP / (TP + FP) | Proportion of positive predictions that are truly positive |
| Recall (Sensitivity) | TP / (TP + FN) | Proportion of actual positives correctly identified |
| Specificity | TN / (TN + FP) | Proportion of actual negatives correctly identified |
| F1-Score | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of Precision and Recall |
| AUC-ROC | Area under the ROC curve | Discriminative ability across all classification thresholds |

> **Where:** TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives

The **Confusion Matrix** decomposes all predictions into True Positives, True Negatives, False Positives, and False Negatives, providing granular insight into classification errors for both disease prediction tasks.

Post-inference, the model's output probability score is mapped to a clinical risk tier as defined in Table 3.2.

**Table 3.2: Risk Stratification Thresholds**

| Predicted Probability (p) | Risk Tier | Clinical Recommendation |
|---|---|---|
| p < 0.40 | Low Risk | Routine health monitoring; no immediate intervention required |
| 0.40 ≤ p < 0.70 | Medium Risk | Lifestyle intervention advised; follow-up screening recommended |
| p ≥ 0.70 | High Risk | Immediate clinical consultation strongly recommended |

[INSERT FIGURE 3.2: Confusion Matrix — for Heart Disease model and Diabetes model]

---

## 3.6 Model Deployment

**FastAPI API Deployment:** The trained models are deployed as a production REST API using FastAPI. Two prediction endpoints are defined — `/predict/heart` and `/predict/diabetes` — each accepting a JSON request body validated against a Pydantic schema. Upon a valid request, the endpoint applies the pre-fitted `StandardScaler`, invokes the corresponding Random Forest classifier, maps the output probability to a risk tier, and returns a structured JSON response containing the predicted risk level, probability score, and clinical recommendation.

**Docker Containerisation:** The FastAPI application is containerised using a `Dockerfile` built on a `python:3.10-slim` base image, with all dependencies installed from `requirements.txt` and serialised model artefacts bundled into the image at build time. The Uvicorn ASGI server is configured as the application runtime.

**Docker Compose Usage:** A `docker-compose.yml` file defines and orchestrates all four services of the deployment stack, configuring port mappings, persistent volume mounts, inter-service networking via an internal bridge network, and environment variable injection. The complete system is launched using the single command `docker compose up`.

**Table 3.4: Docker Services**

| Service | Port | Purpose |
|---|---|---|
| `cdss-api` | 8000 | FastAPI prediction service |
| `mlflow-server` | 5000 | Experiment tracking and model registry UI |
| `prometheus` | 9090 | Metrics collection and time-series storage |
| `grafana` | 3000 | Real-time monitoring dashboard |

---

## 3.7 Monitoring System

The monitoring infrastructure provides continuous operational visibility into the health and performance of the deployed CDSS. The `prometheus-fastapi-instrumentator` library is integrated into the FastAPI application, automatically exposing a `/metrics` endpoint that Prometheus scrapes at fifteen-second intervals.

**Prometheus Metrics Collection:** Collected metrics include total HTTP request counts partitioned by endpoint path, method, and status code; request duration histograms enabling p50, p90, and p99 latency percentile computation; and per-model prediction event counters for both the heart disease and diabetes endpoints.

**Grafana Dashboard Visualisation:** Grafana is provisioned with a Prometheus data source and a pre-configured dashboard displaying real-time panels for API request throughput, response latency percentiles, prediction volume per model, and error rates.

**API Request Tracking and Latency Monitoring:** All incoming requests are automatically counted and labelled, enabling detection of traffic anomalies. Sustained increases in p99 latency serve as leading indicators of performance regression or resource contention.

**CPU and Memory Monitoring:** Container-level resource utilisation is tracked to ensure the system operates within acceptable computational bounds and to inform scaling decisions for production environments.

---

## 3.8 Prediction Workflow

The end-to-end prediction lifecycle of the CDSS proceeds through the following sequential stages. 

**User Input:** The end user enters patient clinical parameters through the frontend web form. Client-side validation ensures field completeness before submission.

**API Request:** The frontend serialises the validated form data as a structured JSON payload and transmits it via HTTP POST to the corresponding FastAPI prediction endpoint.

**Model Prediction:** The API validates the payload against the Pydantic input schema, applies the pre-fitted `StandardScaler` transformation, and passes the normalised feature vector to the loaded Random Forest classifier to obtain a predicted class label and probability score via `predict_proba()`.

**Risk Classification:** The output probability is evaluated against the defined stratification thresholds (Table 3.2) to assign the patient to the appropriate clinical risk tier — Low, Medium, or High Risk.

**Response Generation:** A structured JSON response is constructed containing the risk tier, probability score, disease type, and clinical recommendation, and returned to the frontend with HTTP 200 status where it is rendered for the end user.

**Monitoring:** Prometheus automatically records the request count increment and end-to-end latency on its next scrape cycle, updating the Grafana monitoring dashboard within fifteen seconds of prediction completion.

[INSERT FIGURE 3.3: Prediction Workflow Flowchart — sequential stages from User Input through API Request, Model Prediction, Risk Classification, Response Generation, to Monitoring update]

---
