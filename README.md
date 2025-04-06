# Car Price Classification - Logistic Regression with Ridge Regularization

![Application Interface](Images/home.png)

## Overview
This project classifies cars into price categories using Logistic Regression with Ridge regularization. It includes custom metric implementations, MLflow experiment tracking, and CI/CD automation.

**🔗 Live Demo:** [Car Price Classifier](https://st124949.ml.brain.cs.ait.ac.th/)

---

## Immediate Access
[![Try Now](https://img.shields.io/badge/TRY%20THE%20MODEL-NOW-brightgreen?style=for-the-badge&logo=azure-pipelines)](https://st124949.ml.brain.cs.ait.ac.th/)

**Direct Model Access:**  
 [https://st124949.ml.brain.cs.ait.ac.th/](https://st124949.ml.brain.cs.ait.ac.th/) 👈

---

##  Features
- 4-class price categorization (0-3)
- Ridge/L2 regularization toggle
- Custom precision/recall/F1 implementations
- MLflow experiment tracking
- CI/CD with GitHub Actions

---

##  Implementation

### Task 3: Deployment
- **MLflow Tracking**: [https://mlflow.cs.ait.ac.th/](https://mlflow.ml.brain.cs.ait.ac.th/)
- **Model Registry**: Registered as `st124949-a3-model`
- **CI/CD Pipeline**:
  - Unit tests for input validation
  - Auto-deploy on successful tests
  - **CI/CD Status**: [![GitHub Actions](https://img.shields.io/badge/CI/CD_Results-PASSING-green?logo=githubactions)](https://github.com/napassornsp/ML_A3_Napassorn/actions)

---

## Usage

1. **Access the Live Model**  
   [https://st124949.ml.brain.cs.ait.ac.th/](https://st124949.ml.brain.cs.ait.ac.th/)

2. **Input Car Details**  
   ![Input Form](Images/inputA3.png)  
   - Manufacturing Year
   - Kilometers Driven
   - Mileage
   - Engine Capacity
   - Max Power

3. **Toggle Ridge Regularization**

4. **Get Instant Prediction**  
   ![Result](Images/resultA3.png)  
   Displays price category and range

---

## MLflow Tracking
![Experiment UI](Images/mlflow_server.png)  
Track model runs and compare metrics on MLflow server.

---

## Model Validation
| Custom Implementation | sklearn Report |
|-----------------------|----------------|
| ![Manual Metrics](Images/manual_classification_report.png) | ![sklearn Metrics](Images/sk_classification_report.png) |

---

## 🔍 CI/CD Results
[![CI/CD History](https://img.shields.io/badge/View_Full_CI/CD_History-HERE-blue?style=flat-square)](https://github.com/napassornsp/ML_A3_Napassorn/actions)  
Monitor automated testing and deployment status through our GitHub Actions pipeline.

---
