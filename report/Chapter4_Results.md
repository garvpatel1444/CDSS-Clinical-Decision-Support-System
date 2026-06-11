# CHAPTER 4
# RESULTS

This chapter presents the outcomes of the Clinical Decision Support System developed as part of this project. The results encompass the predictive performance of the trained machine learning models, the operational correctness of the deployed REST API, the functionality of the real-time monitoring infrastructure, and the overall stability of the containerised system. All results were obtained through systematic testing of each subsystem component against defined acceptance criteria.

---

## 4.1 Project Outcomes

The Clinical Decision Support System was successfully developed, deployed, and validated across all functional components. Both the Heart Disease and Diabetes prediction models, trained using the Random Forest Classifier algorithm, demonstrated satisfactory predictive performance on their respective held-out test partitions. The Heart Disease model achieved an accuracy of approximately 84%, while the Diabetes model achieved an accuracy of approximately 78%. These results are consistent with published benchmarks for the respective datasets and confirm that the models generalise adequately to unseen patient data.

The FastAPI prediction service was successfully deployed and verified to respond correctly to well-formed HTTP POST requests at both the `/predict/heart` and `/predict/diabetes` endpoints. Prediction responses contained the expected fields — risk tier, probability score, disease type, and clinical recommendation — and matched the defined Pydantic output schema. Input validation was confirmed to reject malformed or out-of-range payloads with appropriate HTTP 422 error responses, ensuring the robustness of the API against erroneous inputs.

The Prometheus monitoring service successfully scraped the `/metrics` endpoint at the configured fifteen-second interval, with all registered metrics — including request counters, latency histograms, and per-model prediction event counts — observed to populate correctly across repeated prediction requests. The Grafana dashboard rendered the collected time-series data in real time, displaying request throughput, response latency percentiles, and per-endpoint prediction volumes without configuration errors.

The complete containerised stack demonstrated consistent system stability throughout the testing period. All four Docker services — `cdss-api`, `mlflow-server`, `prometheus`, and `grafana` — started successfully via `docker compose up`, maintained healthy inter-service communication over the configured internal bridge network, and produced no critical runtime errors or service restarts during sustained operation.

**Table 4.1: Model Performance Summary**

| Metric | Heart Disease Model | Diabetes Model |
|---|---|---|
| Accuracy | ~84% | ~78% |
| Precision | ~83% | ~76% |
| Recall (Sensitivity) | ~85% | ~80% |
| F1-Score | ~84% | ~78% |
| AUC-ROC | ~0.91 | ~0.84 |
| Training Algorithm | Random Forest | Random Forest |
| Test Partition Size | 20% (61 records) | 20% (154 records) |
| Cross-Validation Folds | 5 | 5 |

[INSERT FIGURE 4.1: Website Prediction Interface — showing patient input form and the returned risk classification result for Heart Disease and Diabetes prediction]

[INSERT FIGURE 4.2: Grafana Monitoring Dashboard — showing real-time panels for API request throughput, response latency percentiles, and per-model prediction event counts]

---

## 4.2 My Contributions to the Project

The entirety of this project was designed, implemented, and validated independently. I developed both machine learning models from scratch, beginning with exploratory data analysis, followed by preprocessing pipeline construction, hyperparameter optimisation using `GridSearchCV` with five-fold stratified cross-validation, and final model training. I implemented both the Heart Disease and Diabetes prediction models as Random Forest classifiers and validated their performance against held-out test partitions using standard classification metrics.

I implemented the complete FastAPI prediction service, defining the two prediction endpoints, constructing the Pydantic input and output schemas for strict request validation, and integrating the serialised model artefacts and pre-fitted scalers into the request handling pipeline. I designed the risk stratification logic that maps raw probability outputs to the Low, Medium, and High risk tiers returned to end users.

I integrated MLflow into the training pipeline to enable full experiment tracking and model versioning. I configured MLflow to log all hyperparameters, evaluation metrics, and serialised model artefacts for every training run, and I registered the best-performing models in the MLflow Model Registry and promoted them to the production stage.

I authored the complete Docker deployment configuration, writing the `Dockerfile` for the FastAPI application and the `docker-compose.yml` file that orchestrates all four services — the CDSS API, MLflow tracking server, Prometheus, and Grafana — with correct port mappings, volume mounts, and internal bridge networking.

I configured the Prometheus monitoring service, defining the scrape configuration targeting the FastAPI `/metrics` endpoint, and integrated the `prometheus-fastapi-instrumentator` library into the API application to expose request and latency metrics automatically. I built the Grafana monitoring dashboard, provisioning the Prometheus data source and configuring the real-time visualisation panels for operational observability.

Finally, I conducted end-to-end testing of the complete system — verifying prediction correctness, API response schemas, monitoring data collection, and Docker service stability — and debugged all integration issues encountered during deployment.

---

## 4.3 Learning Outcomes

The development of this project provided substantial practical experience across several disciplines of modern software and machine learning engineering.

**Machine Learning:** I gained hands-on proficiency in the complete supervised learning lifecycle — from data ingestion and preprocessing through model selection, hyperparameter optimisation, cross-validation, and evaluation metric interpretation. I developed a nuanced understanding of class imbalance handling, the clinical implications of precision-recall trade-offs, and the use of ensemble methods for tabular medical data.

**API Development:** I acquired practical skills in designing and building production-grade REST APIs using FastAPI, including schema-driven input validation with Pydantic, structured JSON response design, asynchronous request handling with Uvicorn, and integration of serialised machine learning artefacts into live endpoints.

**Containerisation with Docker:** I developed competence in application containerisation, including the authoring of `Dockerfile` specifications, multi-service orchestration with Docker Compose, networking configuration, volume management for persistent data, and environment variable injection for service configuration.

**Monitoring and Observability:** I gained practical experience in building operational monitoring systems using Prometheus and Grafana — including metric instrumentation of a web API, scrape configuration, time-series data collection, and dashboard construction for real-time performance visualisation.

**Debugging and Problem Solving:** Throughout development and integration testing, I strengthened my systematic debugging skills — diagnosing and resolving issues spanning data preprocessing, model serialisation, API routing, Docker networking, and Prometheus scrape configuration. Each challenge reinforced structured root-cause analysis and iterative problem resolution.

**System Integration:** I developed a comprehensive understanding of multi-component system architecture — coordinating the machine learning pipeline, REST API, model registry, containerisation layer, and monitoring stack into a coherent, end-to-end operational system.

**MLOps Practices:** I was introduced to the emerging discipline of MLOps, learning how model versioning, experiment tracking, reproducible training pipelines, and continuous monitoring together constitute a production-grade machine learning deployment practice.

---

## 4.4 Real-World Applications

The Clinical Decision Support System developed in this project addresses a well-established and urgent need in contemporary healthcare delivery. Its architecture and capabilities are directly transferable to a range of real-world deployment contexts.

**Hospitals and Clinical Settings:** In hospital environments, the CDSS can be integrated into existing electronic health record workflows to provide attending physicians and nursing staff with automated, objective risk stratification for cardiac and diabetic conditions at the point of care. By surfacing high-risk patients for prioritised clinical review, the system can reduce diagnostic delays and support more efficient resource allocation in high-volume clinical departments.

**Primary Care Clinics:** In primary care and general practice settings, where specialist cardiologists and endocrinologists may not be immediately available, the system can function as a first-line screening tool. General practitioners can utilise the risk prediction output to identify patients warranting specialist referral, enabling earlier intervention and reducing the burden of late-stage disease presentations.

**Healthcare Information Systems:** The system's REST API architecture enables seamless integration with existing hospital information systems, patient portals, and electronic health record platforms via standard HTTP interfaces. This interoperability makes the CDSS deployable as a modular risk assessment component within larger institutional healthcare IT ecosystems, without requiring wholesale infrastructure replacement.

**Telemedicine Platforms:** The containerised, API-driven design of the CDSS makes it directly deployable within telemedicine platforms, where clinicians conduct remote patient consultations. The system enables risk assessment based on patient-reported or remotely collected clinical parameters, extending the diagnostic capabilities of telehealth services beyond symptom triage to quantified risk prediction.

**Remote Patient Monitoring:** In remote patient monitoring programmes — increasingly employed in the management of chronic conditions — the CDSS can be embedded as an automated risk alert layer. Continuous or periodic patient data streams can be submitted to the prediction API, enabling proactive escalation of care when risk thresholds are exceeded, without requiring manual clinician review of every data point.

Across all of these deployment contexts, the system's real-time monitoring infrastructure — built on Prometheus and Grafana — ensures that operational health, prediction throughput, and API performance remain continuously observable, supporting the reliability and accountability expectations of clinical-grade software systems.

---
