# 🛡️ CyberGuard AI 2.0 - Enterprise SOC Anomaly Detection & Threat Intelligence Platform

[![Version](https://img.shields.io/badge/version-2.0-blue.svg)](README.md)
[![Developer](https://img.shields.io/badge/developer-Khushi%20Singh-purple.svg)](README.md)
[![Institution](https://img.shields.io/badge/institution-VIT%20Bhopal-orange.svg)](README.md)
[![Year](https://img.shields.io/badge/year-2027-green.svg)](README.md)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Build Status](https://img.shields.io/badge/tests-53%2F53%20passed-success.svg)](#-unit-testing--verification)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-brightgreen.svg?logo=streamlit)](https://cyberguard-ai-gwvp3mvphqfhmf2tebzhzn.streamlit.app/)

> 🚀 **Live Production Dashboard**: [https://cyberguard-ai-gwvp3mvphqfhmf2tebzhzn.streamlit.app/](https://cyberguard-ai-gwvp3mvphqfhmf2tebzhzn.streamlit.app/)

> **CyberGuard AI 2.0** is an end-to-end, enterprise-grade AI security platform for real-time authentication log anomaly detection, multi-class cyber attack classification, explainable risk scoring, SHAP attribution modeling, and an interactive Security Operations Center (SOC) web dashboard powered by a fact-grounded AI Security Copilot.

---

## 📋 Project Metadata
- **Project Name**: CyberGuard AI 2.0
- **Version**: Version 2.0
- **Developer**: Khushi Singh
- **Institution**: VIT Bhopal
- **Release Year**: 2027

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Platform Architecture](#-platform-architecture)
3. [Folder Structure](#-folder-structure)
4. [Installation & Setup](#-installation--setup)
5. [How to Run (CLI & Pipeline)](#-how-to-run-cli--pipeline)
6. [Dataset & Attack Injection Matrix](#-dataset--attack-injection-matrix)
7. [Machine Learning Pipeline](#-machine-learning-pipeline)
8. [Streamlit SOC Dashboard Features](#-streamlit-soc-dashboard-features)
9. [AI Security Copilot](#-ai-security-copilot)
10. [Screenshots & Visualizations](#-screenshots--visualizations)
11. [Future Scope](#-future-scope)
12. [Authors & Citation](#-authors--citation)

---

## 🛡️ Project Overview

Modern enterprise authentication infrastructure processes millions of telemetry events daily. Sophisticated adversaries bypass traditional static rule-based firewalls using credential stuffing, impossible travel, and device spoofing techniques.

**CyberGuard AI 2.0** solves this challenge by implementing a hybrid unsupervised-supervised machine learning architecture:
- **Unsupervised Anomaly Detection**: Learns baseline user behavior purely on normal telemetry events (`is_anomaly == 0`) using **Isolation Forest** ($99.34\%$ Precision, $100.0\%$ Recall, $0.9998$ ROC AUC).
- **Multi-Class Attack Classification**: Classifies detected anomalies into 6 threat categories using **Random Forest / XGBoost** ($98.83\%$ Accuracy).
- **Explainable Risk Engine**: Computes explainable risk scores ($0–100$) mapped into `Low`, `Medium`, `High`, and `Critical` severity tiers.
- **SHAP Explainability**: Translates complex Shapley feature attributions into plain-English natural language SOC analyst summaries.
- **AI Security Copilot**: Ground-truth natural language assistant for alert investigation and SOC response playbooks.

---

## 🏗️ Platform Architecture

```
                                  +---------------------------------------+
                                  |   Module 1: Baseline Log Generator    |
                                  |   (100,000 Records, 30 Days Span)     |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |   Module 2: Cyber Attack Injector     |
                                  |   (3.0% Attack Ratio across 6 Types)  |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  | Module 3: Feature Engineering Engine  |
                                  | (44 ML Features & StandardScaler)     |
                                  +-------------------+-------------------+
                                                      |
                                        +-------------+-------------+
                                        |                           |
                                        v                           v
                      +----------------------------------+ +-----------------------------------+
                      | Module 4: Isolation Forest       | | Module 5: Multi-Class Classifier  |
                      | (Trained ONLY on Normal Data)    | | (Trained ONLY on Anomalies)      |
                      +-----------------+----------------+ +----------------+------------------+
                                        |                                   |
                                        +-------------+-------------+-------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  | Module 6: Multi-Factor Risk Engine    |
                                  | (Explainable Score 0 - 100)           |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  | Module 7: SHAP Explainability Engine  |
                                  | (Natural Language Threat Explanations)|
                                  +-------------------+-------------------+
                                                      |
                                        +-------------+-------------+
                                        |                           |
                                        v                           v
                      +----------------------------------+ +-----------------------------------+
                      | Module 8: Enterprise Dashboard   | | Module 9: AI Security Copilot    |
                      | (10 Pages, Plotly Dark SOC UI)   | | (Fact-Grounded Natural Language) |
                      +----------------------------------+ +-----------------------------------+
```

---

## 📁 Folder Structure

```
CyberGuardAI 2.0/
├── config.py                       # Centralized configuration singleton
├── requirements.txt                # Production Python package dependencies
├── README.md                       # Master project documentation
├── hackathon_presentation.md       # 10-Slide Hackathon Pitch Deck
├── data/
│   ├── raw/                        # Baseline raw telemetry logs
│   ├── processed/                  # Attack-injected & feature-engineered datasets
│   └── predictions/                # Anomaly, classification, risk, and SHAP reports
├── saved_models/
│   ├── feature_scaler.joblib       # Fitted StandardScaler artifact
│   ├── isolation_forest.joblib     # Trained Isolation Forest model artifact
│   └── attack_classifier.joblib   # Trained Multi-Class Attack Classifier artifact
├── logs/                           # System operational execution logs
├── reports/
│   └── figures/                    # Diagnostic ROC, PR, Confusion Matrix & SHAP plots
├── src/
│   ├── data/
│   │   ├── log_generator.py        # Baseline synthetic log generator
│   │   ├── attack_injector.py      # Multi-class cyber attack injector
│   │   └── feature_engineering.py  # 44 ML feature extraction engine
│   ├── models/
│   │   ├── anomaly_detector.py     # Isolation Forest anomaly detector
│   │   ├── attack_classifier.py    # Multi-class attack classifier
│   │   ├── risk_engine.py          # Multi-factor risk scoring engine (0-100)
│   │   └── explainable_ai.py       # SHAP TreeExplainer & NL generator
│   ├── copilot/
│   │   └── sec_copilot.py          # AI Security Copilot grounded engine
│   ├── dashboard/
│   │   ├── app.py                  # Main Streamlit web application
│   │   ├── utils.py                # Dark SOC CSS design system & Plotly builders
│   │   ├── simulator.py            # Real-time 1-second telemetry event simulator
│   │   └── pages/                  # 10 Modular Streamlit SOC pages
│   └── utils/
│       └── logger.py               # Centralized logging setup
├── scripts/
│   ├── generate_logs.py            # CLI: Generate baseline logs
│   ├── inject_attacks.py           # CLI: Inject cyber attacks
│   ├── engineer_features.py        # CLI: Extract engineered features
│   ├── train_anomaly_detector.py   # CLI: Train Isolation Forest model
│   ├── train_attack_classifier.py  # CLI: Train Multi-Class Classifier
│   ├── calculate_risk_scores.py    # CLI: Calculate risk scores
│   ├── explain_threats.py          # CLI: Generate SHAP explanations
│   ├── query_copilot.py            # CLI: Query AI Security Copilot
│   └── run_dashboard.py            # CLI: Launch Streamlit SOC Dashboard
└── tests/
    ├── test_log_generator.py       # Unit tests: Baseline log generator
    ├── test_attack_injector.py     # Unit tests: Cyber attack injector
    ├── test_feature_engineering.py # Unit tests: Feature engineering engine
    ├── test_anomaly_detector.py    # Unit tests: Behavioral anomaly detector
    ├── test_attack_classifier.py   # Unit tests: Multi-class attack classifier
    ├── test_risk_engine.py         # Unit tests: Explainable risk engine
    ├── test_explainable_ai.py      # Unit tests: SHAP explainable AI engine
    ├── test_dashboard.py           # Unit tests: Streamlit dashboard loaders & imports
    └── test_copilot.py             # Unit tests: AI Security Copilot intent engine
```

---

## ⚡ Installation & Setup

### Prerequisites
- Python 3.9, 3.10, or 3.11
- Git

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-org/CyberGuardAI-2.0.git
cd CyberGuardAI-2.0

# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Run (CLI & Pipeline)

You can execute the entire pipeline end-to-end using individual CLI scripts or launch the web dashboard directly:

### Step 1: Generate Synthetic Telemetry Logs
```bash
python scripts/generate_logs.py --output data/raw/synthetic_login_logs.csv --num-records 100000
```

### Step 2: Inject Cyber Attacks (3.0% Ratio)
```bash
python scripts/inject_attacks.py --input data/raw/synthetic_login_logs.csv --output data/processed/synthetic_login_logs_with_attacks.csv --attack-ratio 0.03
```

### Step 3: Extract Engineered Features
```bash
python scripts/engineer_features.py --input data/processed/synthetic_login_logs_with_attacks.csv --output data/processed/engineered_features.csv --scaler-output saved_models/feature_scaler.joblib
```

### Step 4: Train Unsupervised Anomaly Detector
```bash
python scripts/train_anomaly_detector.py --input data/processed/engineered_features.csv --model-output saved_models/isolation_forest.joblib --predictions-output data/predictions/anomaly_predictions.csv --reports-dir reports/figures
```

### Step 5: Train Multi-Class Attack Classifier
```bash
python scripts/train_attack_classifier.py --input data/predictions/anomaly_predictions.csv --model-output saved_models/attack_classifier.joblib --predictions-output data/predictions/attack_classification_predictions.csv --reports-dir reports/figures
```

### Step 6: Calculate Explainable Risk Scores
```bash
python scripts/calculate_risk_scores.py --input data/predictions/attack_classification_predictions.csv --output data/predictions/risk_scoring_results.csv
```

### Step 7: Compute SHAP Explanations
```bash
python scripts/explain_threats.py --input data/predictions/risk_scoring_results.csv --model saved_models/isolation_forest.joblib --output data/predictions/explainable_anomaly_reports.csv --reports-dir reports/figures
```

### Step 8: Launch Streamlit Web Dashboard
```bash
python scripts/run_dashboard.py
```
> Access live dashboard at **`http://localhost:8501`**

### Step 9: Query AI Security Copilot via CLI
```bash
python scripts/query_copilot.py --query "Why was User USR-00923 flagged?"
python scripts/query_copilot.py --query "Show all impossible travel attacks"
python scripts/query_copilot.py --query "Which users have the highest risk?"
```

---

## 📊 Dataset & Attack Injection Matrix

The platform generates 100,000 synthetic authentication logs across a 30-day window, maintaining persistent user profiles and device fingerprints, and injecting 3,000 multi-class attack scenarios (3.0% ratio):

| Attack Vector Category | Target Ratio | Primary Indicators & Behavioral Manifestations |
| :--- | :--- | :--- |
| **Brute Force** | $0.5\%$ ($500$ events) | Rapid burst of failed logins ($>5$ failures/hour) from a single IP address. |
| **Credential Stuffing** | $0.5\%$ ($500$ events) | Single IP attempting authentication across multiple distinct user accounts. |
| **Impossible Travel** | $0.5\%$ ($500$ events) | Consecutive logins with calculated physical speed $>900\,\text{km/h}$ between locations. |
| **Device Spoofing** | $0.5\%$ ($500$ events) | Unrecognized device fingerprint, headless browser, or conflicting User-Agent. |
| **Lateral Movement** | $0.5\%$ ($500$ events) | Rapid sequential access across internal corporate subnets and endpoints. |
| **Insider Threat** | $0.5\%$ ($500$ events) | Off-hours access ($00:00–05:00$) to restricted endpoints (`/admin/settings`, `/dev/git-repository`). |

---

## 🤖 Machine Learning Pipeline

```
+-----------------------------------------------------------------------------------+
|                            44 Engineered Features                                 |
|  (Cyclical Temporal, Rolling 1h/24h Frequencies, Novelty Flags, Velocity Speed)  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|               Isolation Forest Anomaly Detector (Unsupervised)                    |
|  • Trained strictly on normal telemetry (is_anomaly == 0)                         |
|  • Decision threshold maps to normalized anomaly_score (0.0 to 1.0)               |
|  • Metrics: Precision 0.9934 | Recall 1.0000 | F1 0.9967 | ROC AUC 0.9998        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|               Multi-Class Attack Classifier (Supervised)                          |
|  • Trained strictly on anomaly telemetry (is_anomaly == 1)                        |
|  • Predicts exact attack category across 6 threat vectors                         |
|  • Metrics: Accuracy 98.83% | Macro F1 0.9884                                     |
+-----------------------------------------------------------------------------------+
```

---

## 🖥️ Streamlit SOC Dashboard Features

The web interface is built with an **Enterprise Dark SOC Theme**, glassmorphism cards, glowing risk badges, and custom Plotly dark charts:

1. **🛡️ Executive Overview**: System KPIs, 30-day threat velocity line chart, risk distribution donut chart, live security alert feed.
2. **📡 Live Login Stream**: Interactive telemetry stream with risk level & attack category filters.
3. **🚨 Alerts & Threat Triage**: Dedicated incident queue for High & Critical events with CSV export.
4. **👤 User Behaviour Analytics (UBA)**: Search user profiles (`USR-XXXXX`), inspect 30-day activity, session duration, and Plotly geographical scatter maps.
5. **⚔️ Attack Analytics**: Multi-class attack vector distribution charts & targeted endpoint heatmaps.
6. **🧮 Risk Scores Engine**: Continuous score distribution ($0–100$) and high-risk user leaderboards.
7. **🧠 SHAP Explainability**: Natural language threat explanation cards & diagnostic SHAP plots.
8. **🤖 AI Security Copilot**: Fact-grounded natural language chatbot answering questions about users, alerts, SHAP attributions, and SOC response playbooks.
9. **📈 Model Performance**: Isolation Forest ROC curve, Precision-Recall curve, and $6 \times 6$ confusion matrices.
10. **⚙️ Settings & Admin**: Threshold sliders & system cache management.

---

## 🤖 AI Security Copilot

The AI Security Copilot provides natural language threat analysis grounded strictly on empirical prediction facts and SHAP attributions with **zero hallucination**:

- **"Why was User USR-00923 flagged?"**: Returns risk level, peak score, anomaly ratio, source IP, country, SHAP explanation string, top feature attributions, and recommended SOC playbooks.
- **"Show all impossible travel attacks"**: Filters dataset by attack category, returning incident counts, affected users, and top incident tables.
- **"Which users have the highest risk?"**: Generates user leaderboard ranked by maximum risk score ($0–100$).
- **"Recommend actions for Brute Force attacks"**: Provides incident response playbooks (Session Revocation, Account Lockout, IP Firewall Blocking, EDR Isolation).

---

## 🧪 Unit Testing & Verification

The codebase includes an extensive **53-test automated unit testing suite** covering all 9 modules:

```bash
python -m unittest discover -s tests
```
```
----------------------------------------------------------------------
Ran 53 tests in 6.025s

OK
```

---

## ✍️ Authors & Citation

**Developed by**: Khushi Singh  
**Institution**: VIT Bhopal  
**Version**: 2.0 (2027)  

```bibtex
@software{cyberguard_ai_2027,
  author = {Khushi Singh},
  title = {CyberGuard AI 2.0: Enterprise SOC Anomaly Detection & Threat Intelligence Platform},
  institution = {VIT Bhopal},
  year = {2027},
  version = {2.0},
  url = {https://github.com/your-org/CyberGuardAI-2.0}
}
```

---
*Built with Python, Scikit-learn, SHAP, Streamlit, and Plotly.*
