# CHAPTER 2: LITERATURE SURVEY

---

## 2.1 Related Work

### 2.1.1 Machine Learning in Healthcare

The application of machine learning (ML) in the healthcare domain has gained considerable momentum over the past two decades, driven by the exponential growth of electronic health records, genomic databases, and clinical imaging datasets. Supervised learning algorithms, in particular, have demonstrated remarkable capability in extracting meaningful predictive patterns from structured clinical data. Obermeyer and Emanuel (2016) argued that ML methods fundamentally differ from traditional biostatistical approaches by their ability to model high-dimensional interactions without requiring explicit feature engineering, making them especially effective in complex medical classification tasks. Studies across oncology, cardiology, and endocrinology have consistently demonstrated that ensemble methods such as Random Forest and gradient boosting achieve classification accuracy comparable to, and in several cases exceeding, that of experienced clinicians when applied to well-curated clinical datasets.

### 2.1.2 Clinical Decision Support Systems

Clinical Decision Support Systems (CDSS) represent a well-established category of health information technology. Early CDSS implementations, such as MYCIN (1970s) and INTERNIST-1, were rule-based expert systems that encoded physician heuristics as logical inference chains. While these systems provided a structured framework for diagnostic reasoning, they were inherently brittle, requiring exhaustive manual curation of rules and failing to generalise beyond their pre-defined knowledge bases. The transition from rule-based systems to data-driven, machine learning-based CDSS marked a paradigm shift in the field. Berner and La Lande (2007) documented this evolution extensively, noting that modern CDSS architectures are distinguished by their capacity to learn from historical patient data, update their predictive models incrementally, and provide probabilistic rather than deterministic outputs — a characteristic that better reflects the inherent uncertainty of clinical diagnosis.

Contemporary CDSS research has expanded to encompass multi-disease prediction frameworks. Numerous studies have demonstrated the feasibility of deploying integrated platforms that simultaneously assess patient risk across multiple conditions using shared infrastructure. The integration of explainability mechanisms — such as SHAP (SHapley Additive exPlanations) values — into CDSS pipelines has further enhanced clinical adoption by enabling physicians to trace model decisions back to specific input features, thereby establishing trust in automated recommendations.

### 2.1.3 Heart Disease Prediction Using Machine Learning

Cardiovascular disease prediction has been one of the most extensively studied problems in biomedical informatics. The UCI Heart Disease Dataset, originally assembled from patient records at the Cleveland Clinic Foundation, has served as a de facto benchmark for this domain. Detrano et al. (1989) first established the clinical relevance of this dataset by identifying thirteen independent variables — including age, sex, chest pain type, resting blood pressure, serum cholesterol, and thalassemia type — as statistically significant predictors of coronary artery disease. Subsequent ML-based studies by Palaniappan and Awang (2008) and Mohan et al. (2019) demonstrated that ensemble classifiers, particularly Random Forest, consistently outperformed single-model approaches including Naïve Bayes, Decision Trees, and k-Nearest Neighbour classifiers on this dataset, achieving accuracy values in the range of 84–92%.

A critical challenge identified across these studies is the management of **class imbalance** — the disproportionate representation of disease-positive and disease-negative cases — which can bias classifiers toward the majority class. The application of `class_weight='balanced'` in the Random Forest configuration, as employed in the present project, directly addresses this limitation by adjusting sample weights inversely proportional to class frequency during training.

### 2.1.4 Diabetes Prediction Using Machine Learning

Diabetes mellitus prediction represents an equally well-researched application domain. The Pima Indians Diabetes Dataset, contributed by Smith et al. (1988) and subsequently adopted by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), has been widely used to benchmark binary classification algorithms. Key features in this dataset — glucose concentration, body mass index, diabetes pedigree function, and age — have been consistently identified as the most discriminative predictors of Type II diabetes onset. Sisodia and Sisodia (2018) conducted a comparative evaluation of Naïve Bayes, Decision Tree, and Support Vector Machine classifiers on this dataset, with Naïve Bayes achieving approximately 76% accuracy. More recent work by Zou et al. (2018) demonstrated that Random Forest classifiers with balanced class weights and tuned hyperparameters can achieve ROC-AUC scores exceeding 0.82 on the same dataset, affirming their suitability for this prediction task.

### 2.1.5 Limitations of Traditional Diagnosis and the Need for Automation

Traditional disease diagnosis relies primarily on physician-led clinical assessment, laboratory investigations, and imaging studies. While these methods remain the gold standard, they exhibit several inherent limitations. First, the **subjectivity of clinical interpretation** introduces inter-physician variability, particularly in ambiguous presentations. Second, the **diagnostic process is time-intensive**, which is particularly problematic in resource-constrained settings where patient volumes are high. Third, **geographic and economic barriers** restrict access to specialist consultation for large segments of the global population, particularly in low- and middle-income countries.

Automated ML-based prediction systems address these limitations by providing **instant, consistent, and data-driven risk assessments** that do not depend on the availability of specialist clinicians. By stratifying patients into Low, Medium, and High risk categories based on probabilistic confidence scores, such systems enable priority-based triage, early preventive intervention, and optimised resource allocation within healthcare systems. Moreover, the deployment of these systems as RESTful APIs and containerised microservices enables seamless integration into existing hospital information systems and telemedicine platforms, extending their reach to underserved populations.

---

## 2.2 Tools and Technologies

### 2.2.1 Python

**Python** is a high-level, dynamically-typed, general-purpose programming language widely regarded as the dominant language for machine learning and data science applications. Its extensive ecosystem of scientific libraries, expressive syntax, and active open-source community make it uniquely suited to rapid prototyping and production deployment of ML systems. In this project, Python serves as the primary implementation language across all components — model training scripts (`train.py`, `train_diabetes.py`), the ML pipeline module (`ml_pipeline/`), and the FastAPI inference service (`app/main.py`).

---

### 2.2.2 Scikit-learn

**Scikit-learn** is an open-source machine learning library for Python, built on top of NumPy and SciPy, providing efficient implementations of a comprehensive range of supervised and unsupervised learning algorithms. It offers a consistent, well-documented API for model training, cross-validation, hyperparameter tuning via `GridSearchCV`, and evaluation using metrics such as accuracy score, classification report, and ROC-AUC score. In this project, Scikit-learn provides the `RandomForestClassifier`, `train_test_split`, `cross_val_score`, `GridSearchCV`, `StandardScaler`, and `classification_report` utilities used throughout the training pipeline. Its `predict_proba()` method is specifically leveraged to generate the continuous probability scores that underpin the three-tier risk classification logic.

---

### 2.2.3 Random Forest Classifier

The **Random Forest Classifier** is an ensemble learning algorithm that constructs a collection of decision trees during training and outputs the class that is the mode of the individual trees' predictions, or the mean probability estimate for probabilistic outputs. It is inherently robust to overfitting due to bagging (bootstrap aggregation) and random feature subsampling at each split. In this project, the Random Forest is trained with 500 estimators, a maximum depth of 10, and balanced class weights to address class imbalance in both datasets. This configuration was selected to maximise probability calibration quality, ensuring that the output scores reliably represent true risk probabilities.

---

### 2.2.4 FastAPI

**FastAPI** is a modern, high-performance Python web framework for building RESTful APIs, based on standard Python type hints and powered by the ASGI server Uvicorn. It offers automatic OpenAPI documentation generation, request validation via Pydantic models, and asynchronous request handling. In this project, FastAPI exposes two prediction endpoints — `POST /predict/heart` and `POST /predict/diabetes` — that accept structured patient input, invoke the respective MLflow-registered models, compute probability scores, classify risk levels, and return structured JSON responses including the prediction, probability, risk level, and medical recommendation. Prometheus instrumentation middleware is also integrated directly into the FastAPI application for automated metric exposure at the `/metrics` endpoint.

---

### 2.2.5 Docker and Docker Compose

**Docker** is an open-source containerisation platform that packages applications and their dependencies into lightweight, portable containers, ensuring consistent execution across diverse computing environments. **Docker Compose** is a tool for defining and running multi-container Docker applications using a declarative YAML configuration. In this project, Docker Compose orchestrates four containerised services: the **FastAPI inference API** (port 8000), the **MLflow tracking server** (port 5000), the **Prometheus monitoring agent** (port 9090), and the **Grafana dashboard** (port 3000). Shared volumes are used to persist MLflow experiment data across container restarts, and inter-service networking is configured via Docker's internal DNS, enabling the API container to resolve the MLflow server by hostname.

---

### 2.2.6 MLflow

**MLflow** is an open-source platform for managing the complete machine learning lifecycle, encompassing experiment tracking, model packaging, model registry, and model serving. It provides a structured framework for logging training parameters, evaluation metrics, and model artifacts, enabling reproducible experimentation and systematic comparison of model variants. In this project, MLflow logs hyperparameters (number of estimators, maximum depth, minimum samples leaf, class weight) and performance metrics (accuracy, ROC-AUC) for each training run. Trained models are registered in the MLflow Model Registry under versioned aliases — `CDSS_Model@production` and `CDSS_Diabetes_Model@production` — and are loaded dynamically by the FastAPI service at startup using `mlflow.pyfunc.load_model()`.

---

### 2.2.7 Prometheus

**Prometheus** is an open-source systems monitoring and alerting toolkit designed for reliability and scalability. It operates on a **pull-based metrics collection model**, where it periodically scrapes HTTP endpoints that expose metrics in a standardised text format. In this project, Prometheus is configured to scrape the CDSS API's `/metrics` endpoint at five-second intervals. Two custom metrics are instrumented within the FastAPI application: `api_request_count` — a Counter tracking the total number of prediction requests per endpoint — and `api_request_latency_seconds` — a Histogram recording the distribution of end-to-end request processing times. These metrics provide real-time visibility into system utilisation and performance characteristics.

---

### 2.2.8 Grafana

**Grafana** is an open-source analytics and visualisation platform that connects to various data sources — including Prometheus — and renders interactive, configurable dashboards for real-time and historical data exploration. It enables the construction of panels displaying time-series graphs, gauges, heatmaps, and statistical summaries without requiring SQL or programming expertise. In this project, Grafana consumes the Prometheus metrics stream to provide a live operational dashboard for the CDSS deployment, visualising API request rates, latency distributions, and endpoint-level traffic patterns. This observability layer enables proactive identification of performance anomalies and informs scaling decisions.

---

### 2.2.9 Git

**Git** is a distributed version control system that tracks changes in source code and enables collaborative development through branching, merging, and commit history management. In this project, Git is used to version-control all project artefacts — including training scripts, the FastAPI application, Dockerfile, Docker Compose configuration, Prometheus configuration, and the ML pipeline module — providing a complete, auditable history of development changes. The `.gitignore` configuration ensures that large binary files (model artefacts, virtual environment dependencies) are excluded from version tracking, keeping the repository lightweight and portable.

---

*End of Chapter 2*
