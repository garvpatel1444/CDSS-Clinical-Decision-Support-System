# CDSS MLOps — Presentation Slide Content
### BTEP Final Presentation | 14 Slides (Template Mapped)

> **📌 Rule:** Max 6 bullets per slide. Speak the details — don't write them.
> `📷` = insert your image here.

---

## SLIDE 1 — Header Slide
> ✅ Pre-filled in template. No changes needed.

---

## SLIDE 2 — Title Slide

| Field | Fill In |
|---|---|
| Student Name | *(Your name)* |
| Title of BTEP | **Clinical Decision Support System (CDSS) Using ML for Heart & Diabetes Risk Prediction with MLOps** |
| Roll No. | *(Your roll number)* |
| Guide / Industry | *(Mentor name & department)* |
| Start Date | *(e.g. January 2026)* |
| End Date | *(e.g. May 2026)* |

---

## SLIDE 3 — Outline of Presentation

| # | Section |
|---|---|
| 1 | Motivation |
| 2 | Literature Survey |
| 3 | Tools & Technologies |
| 4 | Objectives |
| 5 | Project Outcomes |
| 6 | Methodology |
| 7 | Results & Architecture |
| 8 | Novelty & Contributions |
| 9 | Gantt Chart |
| 10 | Who Will Benefit |

---

## SLIDE 4 — Motivation

**Heading:** Motivation

- Heart disease causes **~17.9 million deaths/year** (WHO)
- **422+ million** people worldwide affected by diabetes
- Early diagnosis is **critical** — but access to specialists is limited
- Manual diagnosis is **slow, inconsistent**, and resource-heavy
- ML can automate accurate risk screening from routine patient data
- Need for a **scalable, real-time, AI-driven** clinical support tool

---

## SLIDE 5 — Literature Survey

**Heading:** Literature Survey

**Section A — Heart Disease Prediction**

| Author(s) | Genuine Finding |
|---|---|
| Detrano et al. (1989) | Applied probabilistic algorithm (logistic regression) on Cleveland dataset (303 patients, 13 features) — achieved ~77% diagnostic accuracy for coronary artery disease |
| Mohan et al. (2019) | Proposed a hybrid Random Forest + Linear Model (HRFLM); achieved **88.7% accuracy** on Cleveland dataset; identified max heart rate, chest pain type & thalassemia as top predictors |

**Section B — Diabetes Prediction**

| Author(s) | Genuine Finding |
|---|---|
| Kavakiotis et al. (2017) | Systematic review of 85 ML papers (2010–2017) on diabetes; **SVM was most frequently used** (75–88% accuracy); Pima Indians Dataset used in ~48% of studies; feature selection and class imbalance handling identified as major challenges |

**Section C — Clinical Decision Support & MLOps**

| Author(s) | Genuine Finding |
|---|---|
| Sculley et al. (2015) | Showed that in real ML systems, **actual model code is a tiny fraction**; most complexity is in surrounding pipelines — data ingestion, feature extraction, monitoring — creating significant "technical debt" |
| Kreuzberger et al. (2022) | Defined MLOps with **9 core principles**: CI/CD automation, workflow orchestration, reproducibility, versioning, continuous training, continuous monitoring, and metadata tracking |
| Obermeyer & Emanuel (2016) | Argued ML can identify patterns in clinical data that classical statistics cannot; also **warned** about algorithmic bias and black-box risks in high-stakes medical decisions |

**Gap identified in existing literature:**
- Most studies are **experimental** — not deployed as live systems
- Outputs are **binary** (disease / no disease) — limited clinical use
- **No integrated MLOps pipeline** (versioning + deployment + monitoring) in CDSS research

---

## SLIDE 6 — Relevant Tools & Technologies

**Heading:** Relevant Tools and Technologies

| Tool | Role |
|---|---|
| Python + Scikit-learn | Model training & evaluation |
| Random Forest | Disease risk classification |
| FastAPI | Real-time prediction REST API |
| Docker & Compose | Containerised, portable deployment |
| MLflow | Experiment tracking & model versioning |
| Prometheus | API metrics collection (every 15 sec) |
| Grafana | Live monitoring dashboard |
| Git | Version control |

---

## SLIDE 7 — Objectives

**Heading:** Objectives of the Project

1. Train ML models for **heart disease & diabetes** prediction
2. Implement **3-tier risk output** — Low / Medium / High
3. Deploy as a **FastAPI REST API** (real-time predictions)
4. **Containerise** all services with Docker & Docker Compose
5. Track models with **MLflow** (versioning + registry)
6. Monitor live system via **Prometheus + Grafana**
7. Deliver a complete **end-to-end MLOps pipeline**

---

## SLIDE 8 — Project Outcomes

**Heading:** Project Outcomes

| Metric | Heart Disease | Diabetes |
|---|---|---|
| Accuracy | **~84%** | **~78%** |
| Recall | ~85% | ~80% |
| F1-Score | ~84% | ~78% |
| AUC-ROC | **~0.91** | **~0.84** |

- ✅ API deployed — returns risk tier + recommendation
- ✅ Docker stack ran stably across all 4 services
- ✅ Prometheus + Grafana confirmed operational

📷 **INSERT: Confusion Matrix (Heart + Diabetes)**
*Caption: "Confusion Matrix — Heart Disease (left) & Diabetes (right)"*

---

## SLIDE 9 — Methodology

**Heading:** Methodology

📷 **INSERT: System Architecture Diagram** ← Main visual
*Caption: "System Architecture — CDSS MLOps Pipeline"*

**Steps (brief, alongside image):**

| Step | What Happened |
|---|---|
| Data | Heart (303 records, 13 features) + Diabetes (768 records, 8 features) |
| Preprocessing | Median imputation, StandardScaler, 80:20 stratified split |
| Training | Random Forest + GridSearchCV (5-fold CV) + MLflow logging |
| Deployment | FastAPI + Docker Compose (4 services) |
| Monitoring | Prometheus scrapes /metrics → Grafana dashboard |

📷 **INSERT (optional): User Flowchart / Prediction Workflow**
*Caption: "Prediction Workflow"*

---

## SLIDE 10 — Results / Findings / Solution / Architecture

**Heading:** Results / Findings / Solution / Architecture

📷 **INSERT: Website Screenshots** (patient form + risk result)
*Caption: "CDSS Web Interface — Input Form & Risk Result"*

📷 **INSERT: Grafana Dashboard**
*Caption: "Grafana Real-Time Monitoring Dashboard"*

**Key results:**
- Heart: **84% accuracy | AUC 0.91**
- Diabetes: **78% accuracy | AUC 0.84**
- Output: Risk level + probability + recommendation

| Probability | Risk | Action |
|---|---|---|
| < 0.30 | 🟢 Low | Maintain healthy lifestyle |
| 0.30–0.70 | 🟡 Medium | Consult doctor |
| ≥ 0.70 | 🔴 High | Immediate consultation |

---

## SLIDE 11 — Novelty / Contributions

**Heading:** Novelty in the Project / Your Contributions

**What's novel:**
- Production-ready CDSS — not just a notebook demo
- 3-tier risk output (Low/Medium/High) — not just binary
- Full MLOps pipeline: versioning + containerisation + monitoring

**My contributions (built independently):**
- 🔬 Both ML models — data to evaluation
- 🌐 FastAPI backend with 2 prediction endpoints
- 📦 MLflow — experiment logging + model registry
- 🐳 Docker + Compose — full multi-service stack
- 📊 Prometheus + Grafana monitoring pipeline
- 🧪 End-to-end system testing

---

## SLIDE 12 — Learning Outcomes

**Heading:** Learning Outcomes for You

| Area | What I Gained |
|---|---|
| Machine Learning | Full supervised learning lifecycle — EDA → tuning → evaluation |
| REST API | FastAPI, Pydantic validation, real-time model inference |
| Docker | Containerisation, multi-service Compose, networking |
| MLOps | Experiment tracking, model versioning, production deployment |
| Monitoring | Prometheus instrumentation, Grafana dashboards |
| Debugging | Root-cause analysis across API, data, Docker, and ML layers |
| System Design | Coordinating 5+ components into one production system |

---

## SLIDE 13 — Gantt Chart

**Heading:** Gantt Chart

📷 **INSERT: Gantt Chart image** (if you have one)

*If no image — use this table:*

| Phase | Task | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|---|
| 1 | Literature Review & Planning | ✅ | | | | |
| 2 | Data Collection & Preprocessing | ✅ | ✅ | | | |
| 3 | Model Training & Evaluation | | ✅ | ✅ | | |
| 4 | FastAPI Development | | | ✅ | | |
| 5 | Docker + MLflow Integration | | | ✅ | ✅ | |
| 6 | Prometheus + Grafana Setup | | | | ✅ | |
| 7 | Testing & Debugging | | | | ✅ | ✅ |
| 8 | Report & Presentation | | | | | ✅ |

---

## SLIDE 14 — Who Will Benefit

**Heading:** Who Will Benefit from Your Work

**Industry / Organisations:**
- 🏥 **Hospitals** — Automated risk screening at point of care
- 🏪 **Primary Care Clinics** — First-line tool where specialists aren't available
- 💻 **Telemedicine Platforms** — Risk prediction for remote consultations
- 📡 **Remote Monitoring Programs** — Automated alerts when thresholds are crossed
- 🏛️ **Government Health Depts.** — Population-level chronic disease screening

**Societal Benefits:**
- Earlier detection → fewer late-stage diagnoses → reduced mortality
- Makes expert-level risk screening **accessible and affordable**
- Scalable, deployable blueprint for responsible AI in healthcare

---

## 📷 IMAGE PLACEMENT SUMMARY

| Slide | Template Title | Image | Caption |
|---|---|---|---|
| **8** | Project Outcomes | Confusion Matrix (Heart + Diabetes) | Confusion Matrix — Heart & Diabetes |
| **9** | Methodology | System Architecture Diagram | System Architecture |
| **9** | Methodology | Flowchart *(optional)* | Prediction Workflow |
| **10** | Results / Architecture | Website Screenshots | CDSS Web Interface |
| **10** | Results / Architecture | Grafana Dashboard | Grafana Monitoring Dashboard |
