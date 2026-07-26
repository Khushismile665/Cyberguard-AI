# 🛡️ CyberGuard AI 2.0 - Hackathon Presentation Pitch Deck

---

## 📌 Slide 1: Title Slide & Project Overview

### **CyberGuard AI 2.0 (Version 2.0)**
#### *Next-Generation Enterprise SOC Anomaly Detection & Threat Intelligence Platform*

```
  [ 🛡️ SHIELD BADGE ]
  ==================================================================
  DEVELOPED BY: Khushi Singh | INSTITUTION: VIT Bhopal (2027)
  AI-POWERED AUTHENTICATION SECURITY | ZERO-HALLUCINATION COPILOT
  INSPECTION ENGINE | 10-PAGE ENTERPRISE SOC DASHBOARD
  ==================================================================
```

- **Developer**: Khushi Singh (VIT Bhopal)
- **Version**: 2.0 (Release 2027)
- **Hackathon Track**: AI/ML for Enterprise Cybersecurity & Threat Intelligence
- **Core Stack**: Python, Scikit-learn, Isolation Forest, XGBoost, SHAP, Streamlit, Plotly
- **Key Highlight**: $99.34\%$ Precision, $100.0\%$ Attack Recall, 10-Page Commercial SOC UI

---

## 📌 Slide 2: Problem Statement

### **The Modern Authentication Cyber Threat Crisis**

> *Enterprise authentication perimeter is under constant siege by automated, distributed threat actors.*

- 📈 **Scale Explosion**: Enterprise IAM systems process **10M+ daily authentication events** across global hybrid networks.
- 🎭 **Sophisticated Attack Vectors**: Attackers bypass multi-factor authentication (MFA) via:
  - **Credential Stuffing** using leaked dark-web database dumps.
  - **Impossible Physical Travel** across disparate geographic countries within minutes.
  - **Device & Fingerprint Spoofing** utilizing headless automated browser scripts.
- ⏱️ **Analyst Fatigue**: SOC analysts face **thousands of daily false-positive alerts**, causing high mean time to detect (MTTD) and mean time to respond (MTTR).

---

## 📌 Slide 3: Existing Issues in Traditional SOC Infrastructure

### **Why Legacy Security Solutions Fail**

```
+--------------------------+-------------------------------------------------------+
| Legacy Limitation        | Real-World Operational Impact                         |
+--------------------------+-------------------------------------------------------+
| 1. Static Rule Triggers  | Hardcoded thresholds (e.g. 5 failed logins) fail to  |
|                          | detect low-and-slow lateral movement attacks.         |
| 2. High False Positives  | Over 85% of security alerts are noise, burying true    |
|                          | critical security breaches.                           |
| 3. Black-Box AI Models   | Deep neural networks flag threats without context,    |
|                          | leaving analysts unable to explain WHY a user failed. |
| 4. Disconnected Tooling  | SIEM, SOAR, and UBA engines operate in silos, requiring|
|                          | manual data stitching during active incident triage.  |
+--------------------------+-------------------------------------------------------+
```

---

## 📌 Slide 4: The Solution - CyberGuard AI 2.0

### **A Modular, Hybrid Unsupervised-Supervised Threat Intelligence Platform**

- 🤖 **Zero-Baseline Anomaly Detection**: Unsupervised **Isolation Forest** trained *strictly on normal telemetry* ($100\%$ attack recall) to detect novel zero-day threats.
- 🎯 **Multi-Class Attack Classifier**: Classifies anomalies into **6 distinct attack categories** (*Brute Force, Credential Stuffing, Impossible Travel, Device Spoofing, Lateral Movement, Insider Threat*).
- 🧮 **Explainable Risk Scoring Engine**: Calculates continuous risk scores ($0–100$) mapped into clear severity tiers (`Low`, `Medium`, `High`, `Critical`).
- 🧠 **SHAP Natural Language Explainability**: Converts Shapley attributions into plain-English SOC analyst explanations.
- 💬 **Fact-Grounded AI Security Copilot**: Interactive assistant answering threat queries with zero hallucination.

---

## 📌 Slide 5: End-to-End System Architecture

```
  [ RAW TELEMETRY ] ---> [ 44 FEATURE EXTRACTION ] ---> [ UNSUPERVISED ANOMALY DETECTOR ]
  100k Authentication     Cyclical Time, Rolling        Isolation Forest (Normal Training)
  Logs (30 Days Span)     Velocity, Device Novelty              |
                                                                | (Flagged Anomalies)
                                                                v
  [ EXECUTIVE DASHBOARD ] <--- [ SHAP & RISK ENGINE ] <--- [ MULTI-CLASS CLASSIFIER ]
  10-Page Plotly SOC UI        Explainable Score (0-100)   XGBoost / Random Forest (98.83%)
  & AI Copilot Chat            Natural Language Explanations
```

---

## 📌 Slide 6: Machine Learning Models & Feature Engineering

### **Dual-Model ML Engine & 44 Engineered Telemetry Features**

#### **1. Feature Engineering Engine (44 Features)**:
- **Temporal**: `login_hour`, `day_of_week`, cyclical sine/cosine hour transformations.
- **Behavioral Velocity**: 1-hour/24-hour rolling login frequencies, failed attempt bursts.
- **Novelty Flags**: `is_new_device`, `is_new_location`, OS/Browser One-Hot encodings.
- **Geographic Distance**: Physical speed calculation ($\text{km/h}$) between consecutive logins.

#### **2. Dual-Model Performance Specs**:
- **Unsupervised Isolation Forest**: Trained ONLY on `is_anomaly == 0`.
  - **Precision**: `99.34%` | **Recall**: `100.0%` | **F1 Score**: `0.9967` | **ROC AUC**: `0.9998`
- **Multi-Class Attack Classifier**: Trained ONLY on `is_anomaly == 1`.
  - **Multi-Class Accuracy**: `98.83%` | **Macro F1 Score**: `0.9884`

---

## 📌 Slide 7: Enterprise SOC Dashboard

### **Commercial-Grade 10-Page Interactive Web Application**

```
+-----------------------------------------------------------------------------------+
|  [🛡️ OVERVIEW]   [📡 LIVE STREAM]   [🚨 ALERTS]   [👤 UBA]   [⚔️ ATTACK ANALYTICS]  |
|  [🧮 RISK ENGINE] [🧠 SHAP XAI]      [🤖 COPILOT]  [📈 MODEL METRICS] [⚙️ SETTINGS] |
+-----------------------------------------------------------------------------------+
```

- 🎨 **Enterprise Aesthetics**: Dark space styling (`#070a12`), glassmorphism card containers, Google Fonts (`Outfit`, `Inter`, `JetBrains Mono`).
- ⚡ **Animated KPI Cards**: Multi-color top border beams, smooth 3D hover transitions.
- 🚨 **Glowing Alert Cards**: `@keyframes pulse-glow` pulsating neon red/orange alert indicators.
- 📊 **Interactive Plotly Visualizations**: Area threat timelines, risk donut charts, user scatter geo maps, and $6 \times 6$ confusion matrices.

---

## 📌 Slide 8: Quantifiable Results & Model Validation

### **Benchmarked Performance Metrics**

```
  ISOLATION FOREST ROC AUC           MULTI-CLASS CLASSIFICATION ACCURACY
  [ 0.9998 / 1.0000 ]                 [ 98.83% / 100.0% ]
  =========================          =========================
  • Precision : 99.34%               • Brute Force       : 99.1% F1
  • Recall    : 100.0%               • Credential Stuff  : 98.5% F1
  • F1 Score  : 0.9967               • Impossible Travel : 99.6% F1
  • Missed    : 0 Attacks            • Device Spoofing   : 98.2% F1
```

- **Zero Missed Threat Attacks**: $100\%$ detection recall across 3,000 injected attack incidents.
- **53 Automated Unit Tests**: $100\%$ test suite pass rate in $5.7$ seconds.

---

## 📌 Slide 9: Key Innovations & Competitive Differentiators

### **What Sets CyberGuard AI 2.0 Apart**

1. 🧠 **SHAP Natural Language Translator**:
   - Converts abstract Shapley values ($\phi_i$) into plain English:
     > *"Risk increased due to impossible travel velocity. User logged in from a new country."*
2. 💬 **Zero-Hallucination AI Security Copilot**:
   - Grounded strictly on empirical dataset records and predictions.
   - Answers analyst queries (*"Why was User USR-00923 flagged?"*, *"Show all impossible travel attacks"*).
3. 🛠️ **Automated Incident Response Playbooks**:
   - Provides tailored SOC mitigation actions (*Session Revocation, Step-up FIDO2 MFA, WAF IP Block, EDR Host Isolation*).

---

## 📌 Slide 10: Project Credits & Live Demo

### **Developed by Khushi Singh | VIT Bhopal (2027)**

- 🖥️ **Live Web Dashboard**: `http://localhost:8501`
- 💻 **Run Test Suite**: `python -m unittest discover -s tests`
- 💬 **CLI Copilot Query**: `python scripts/query_copilot.py --query "Why was User USR-00923 flagged?"`
