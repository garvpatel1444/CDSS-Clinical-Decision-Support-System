# CHAPTER 2
# LITERATURE SURVEY

The development of machine learning-based clinical decision support systems is grounded in a rich body of research spanning healthcare informatics, predictive analytics, and artificial intelligence. This chapter reviews the existing literature pertaining to disease prediction systems, the evolution of clinical decision support, and the application of machine learning in medical diagnostics. Additionally, it provides a technical overview of the tools and technologies employed in the development of the proposed system, with justification for their selection based on established academic and industry precedent.

---

## 2.1 Related Work

The application of machine learning to healthcare has been a subject of intensive research over the past two decades. The emergence of large-scale electronic health records (EHRs), digitised medical datasets, and advanced computational infrastructure has collectively enabled the development of data-driven models capable of assisting clinicians in diagnosing, monitoring, and predicting a wide range of medical conditions. Within this domain, cardiovascular disease and diabetes mellitus have received particular scholarly attention owing to their global prevalence, clinical complexity, and the severe consequences of late-stage diagnosis.

Traditional methods of diagnosing heart disease and diabetes have historically relied upon a combination of physical examination, laboratory investigations, imaging studies, and the interpretive judgment of experienced clinicians. While these approaches remain clinically valid, they are inherently subject to several limitations. Manual diagnostic processes are time-consuming, resource-intensive, and susceptible to inter-clinician variability — a phenomenon in which different practitioners may arrive at divergent conclusions when presented with identical patient data. Moreover, in regions where specialist access is constrained, patients frequently present at advanced stages of disease, significantly diminishing the prospects for effective intervention. These structural limitations of conventional diagnosis underline the demand for automated, objective, and scalable clinical tools.

Research into Clinical Decision Support Systems (CDSS) dates to the early 1970s, with foundational systems such as MYCIN — a rule-based expert system for diagnosing bacterial infections — demonstrating the viability of computerised medical reasoning. However, early rule-based CDSS were constrained by the need for exhaustive, manually encoded knowledge bases and their inability to generalise beyond pre-defined clinical rules. The advent of machine learning offered a paradigm shift, enabling systems to derive predictive rules autonomously from historical patient data without explicit programming of domain logic.

Numerous studies have demonstrated the efficacy of supervised learning algorithms for disease classification tasks. Detrano et al. established early benchmarks for the automated classification of coronary artery disease using the Cleveland Heart Disease Dataset, which has since become a standard benchmark in the field. Subsequent work by Aha and Kibler (1991) and, more recently, by Mohan et al. (2019) demonstrated that ensemble methods — particularly Random Forest — consistently outperform single-classifier approaches on heart disease datasets by virtue of their variance reduction properties and robustness to feature noise.

In the domain of diabetes prediction, the Pima Indians Diabetes Dataset, introduced by Smith et al. (1988) and maintained by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), has served as a canonical benchmark for evaluating classification models. Studies by Kavakiotis et al. (2017) conducted a systematic review of machine learning applications in diabetes research and reported that ensemble classifiers and support vector machines consistently achieved superior sensitivity and specificity compared to logistic regression and naive Bayes approaches. Their findings further highlighted the importance of feature selection and class imbalance handling in achieving clinically meaningful predictive performance.

The concept of risk stratification — classifying patients into ordered risk tiers rather than providing binary diagnostic outcomes — has gained prominence as a more clinically informative paradigm. Goldstein et al. (2017) argued that probabilistic risk scores, when mapped to interpretable tiers such as low, medium, and high risk, significantly enhance clinical utility by enabling prioritised patient management and reducing both over-treatment and under-treatment. This three-tier approach forms a cornerstone of the prediction architecture in the present project.

Beyond predictive accuracy, recent scholarship has increasingly emphasised the operational dimension of deploying machine learning models in clinical settings. The discipline of MLOps — an amalgamation of machine learning and software development operations — addresses the challenges of model reproducibility, version control, scalable deployment, and continuous performance monitoring. Sculley et al. (2015), in their seminal paper on hidden technical debt in machine learning systems, identified the gap between model development and reliable production deployment as a critical and frequently underestimated engineering challenge. Subsequent work by Kreuzberger et al. (2022) formalised the MLOps lifecycle, demonstrating that systematic experiment tracking, containerised deployment, and real-time monitoring are essential components of production-ready machine learning systems. The present project directly addresses these requirements through the integration of MLflow, Docker, Prometheus, and Grafana within a cohesive MLOps pipeline.

The limitations identified in the reviewed literature — including over-reliance on single-model architectures, insufficient attention to deployment and monitoring, and the absence of risk stratification in many proposed systems — have directly informed the design decisions of the proposed CDSS. By combining ensemble machine learning with a structured MLOps pipeline and a three-tier risk classification framework, this project seeks to advance beyond proof-of-concept implementations and deliver a system that is both predictively accurate and operationally viable in real-world healthcare contexts.

---

## 2.2 Tools and Technologies

The following section presents a technical overview of the tools and technologies employed in the design, development, and deployment of the Clinical Decision Support System. Each tool is described in terms of its nature, rationale for selection, and specific role within the project.

---

**Python**

Python is a high-level, interpreted, general-purpose programming language widely adopted in scientific computing, data science, and machine learning owing to its expressive syntax, extensive standard library, and a rich ecosystem of third-party packages. In this project, Python serves as the primary development language across all system components — including data preprocessing, model training, API development, and pipeline orchestration. Its compatibility with leading machine learning and web frameworks makes it the unambiguous choice for an integrated healthcare ML system.

---

**Scikit-learn**

Scikit-learn is an open-source machine learning library for Python, built upon NumPy, SciPy, and Matplotlib. It provides a consistent, well-documented API for a broad range of supervised and unsupervised learning algorithms, along with utilities for model evaluation, preprocessing, and cross-validation. In this project, Scikit-learn is used to implement the Random Forest classifier for both heart disease and diabetes prediction models, perform train-test splitting and cross-validation, compute evaluation metrics including accuracy, precision, recall, F1-score, and AUC-ROC, and execute feature scaling and preprocessing transformations.

---

**Random Forest Classifier**

Random Forest is a supervised ensemble learning algorithm that operates by constructing a large number of decision trees during training and aggregating their individual predictions through majority voting (for classification) to produce a final output. The algorithm introduces randomness at two levels — in the sampling of training data (bootstrap aggregation, or bagging) and in the selection of features considered at each split — thereby reducing model variance without materially increasing bias. Random Forest was selected for this project due to its strong empirical performance on tabular clinical datasets, its inherent resilience to overfitting, and its capacity to provide feature importance rankings, which are of direct clinical interpretability value.

---

**FastAPI**

FastAPI is a modern, high-performance web framework for building APIs with Python, based on standard Python type hints and the OpenAPI specification. It is designed for asynchronous request handling and delivers among the highest throughput benchmarks of any Python web framework. In this project, FastAPI is used to expose RESTful prediction endpoints for both heart disease and diabetes risk classification. Patient feature data is submitted to the API in JSON format, processed through the loaded machine learning models, and a structured risk classification response — including the predicted risk tier and associated probability — is returned to the client in real time.

---

**Docker**

Docker is an open-source containerisation platform that enables developers to package applications and their dependencies into portable, self-contained units known as containers. Containers share the host operating system kernel, making them significantly more lightweight than traditional virtual machines while providing equivalent process isolation. In this project, Docker is used to containerise the FastAPI prediction service, the MLflow tracking server, and the Prometheus and Grafana monitoring stack. A Docker Compose configuration file orchestrates the launch and networking of all containers as a unified multi-service deployment, ensuring environmental consistency across development and production.

---

**MLflow**

MLflow is an open-source platform for managing the end-to-end machine learning lifecycle, including experiment tracking, model packaging, and model registry. It provides a structured mechanism for logging parameters, metrics, and artefacts associated with each training run, enabling systematic comparison of experimental outcomes and reproducible model development. In this project, MLflow is used to log all training runs for both the heart disease and diabetes models, store serialised model artefacts, and register production-candidate model versions within the MLflow Model Registry, facilitating controlled and auditable model promotion.

---

**Prometheus**

Prometheus is an open-source systems monitoring and alerting toolkit, originally developed at SoundCloud, and now a graduated project of the Cloud Native Computing Foundation (CNCF). It operates on a pull-based metrics collection model, periodically scraping instrumented application endpoints to collect time-series performance data. In this project, Prometheus is integrated with the FastAPI application via the `prometheus-fastapi-instrumentator` library to collect runtime metrics including API request counts, response latency distributions, and prediction endpoint activity, forming the observability backbone of the deployed system.

---

**Grafana**

Grafana is an open-source analytics and interactive visualisation platform that connects to a wide range of data sources, including Prometheus, to render real-time dashboards and alerting interfaces. In this project, Grafana is configured to connect to the Prometheus metrics store and display a custom monitoring dashboard encompassing API request rate, prediction latency, model inference time distributions, and system resource utilisation. This dashboard provides operational stakeholders with continuous visibility into the health and performance of the deployed CDSS.

---

**Git**

Git is a distributed version control system designed for tracking changes in source code during software development. It enables multiple contributors to collaborate on a codebase, maintains a complete history of modifications, and supports branching and merging workflows for parallel feature development. In this project, Git is used to version-control all source code, configuration files, training scripts, and pipeline definitions, ensuring a complete and auditable development history and enabling reproducible reconstruction of any prior project state.

---

[INSERT TABLE 2.1: Summary of Tools and Technologies — columns: Tool, Category, Version Used, Role in Project]
