# ABSTRACT

Cardiovascular disease and diabetes mellitus represent two of the most prevalent and life-threatening chronic conditions worldwide, yet timely clinical intervention remains constrained by limited access to diagnostic expertise. This project presents the design and implementation of a **Clinical Decision Support System (CDSS)** that leverages machine learning to predict the risk of heart disease and diabetes, thereby augmenting clinical decision-making in healthcare settings.

The system employs a **Random Forest Classifier**, trained independently on the UCI Heart Disease Dataset and the Pima Indians Diabetes Dataset, to generate probabilistic risk scores and classify patient risk into three actionable tiers — **Low**, **Medium**, and **High** — accompanied by context-specific medical recommendations. The trained models are version-controlled and tracked using **MLflow**, enabling reproducible experimentation and seamless model governance.

The inference engine is exposed as a **RESTful API** built with **FastAPI** and containerised using **Docker** and **Docker Compose**, ensuring portability and environment-consistent deployment. Operational observability is achieved through **Prometheus** for real-time metrics collection and **Grafana** for interactive dashboard visualisation, establishing a robust **MLOps** pipeline from training to production monitoring. The proposed system demonstrates the practical feasibility of integrating machine learning into scalable, production-ready healthcare decision support infrastructure.

---
**Word Count: ~175 words**
