# Insurance Claims Fraud Detection System

## Overview

The **Insurance Claims Fraud Detection System** is a machine learning-based application designed to identify potentially fraudulent insurance claims, assess their risk levels, and support faster, more efficient claim processing.

The system uses historical insurance claim data to train machine learning models that distinguish between genuine and potentially fraudulent claims. It incorporates data preprocessing, feature engineering, model evaluation, real-time prediction, configurable risk thresholds, and an interactive analytics dashboard.

The proposed solution uses **LightGBM** as the primary fraud detection model due to its training efficiency, scalability, and ability to handle structured datasets.

## Problem Statement

Insurance companies face significant financial losses due to fraudulent claims. Traditional rule-based detection systems may fail to identify evolving fraud patterns and can generate excessive false positives, increasing manual investigation efforts.

This project addresses these challenges by implementing a data-driven fraud detection system that:

- Identifies potentially fraudulent insurance claims.
- Reduces missed fraud cases and unnecessary investigations.
- Generates risk scores to support claim prioritization.
- Provides analytical insights into fraud patterns.
- Supports scalable claim analysis through a REST API.

## Objectives

- Develop a machine learning model to classify insurance claims as fraudulent or genuine.
- Handle class imbalance in historical claims data.
- Compare multiple classification algorithms.
- Evaluate model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
- Provide real-time predictions through a Flask API.
- Visualize model performance, claim distributions, and prediction statistics.
- Enable configurable decision thresholds and prediction logging.

## Key Features

### 1. Machine Learning-Based Fraud Detection
Uses trained classification models to analyze insurance claim attributes and identify suspicious patterns.

### 2. Model Comparison and Selection
Evaluates multiple machine learning algorithms to select a suitable model for fraud detection.

### 3. Risk Scoring
Returns a fraud prediction, model confidence or probability, and a risk category to support investigation prioritization.

### 4. Adjustable Decision Threshold
Allows the prediction threshold to be configured according to operational requirements and the trade-off between missed fraud and false alarms.

### 5. Interactive Analytics Dashboard
Displays classification metrics, fraud distribution, feature importance, and prediction statistics.

### 6. Prediction Logging
Records prediction results in CSV format for monitoring and subsequent analysis.

### 7. REST API Integration
Exposes prediction and dashboard data through Flask endpoints for integration with a frontend application.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, LightGBM, XGBoost |
| Class Imbalance Handling | SMOTE |
| Data Preprocessing | ColumnTransformer, OneHotEncoder, StandardScaler |
| Model Evaluation | Accuracy, Precision, Recall, F1-score, ROC-AUC |
| Backend | Flask, REST API |
| Frontend | React.js |
| Visualization | Chart.js or Recharts |
| Model Persistence | Joblib |
| Dataset | Historical insurance claims dataset |
| Version Control | Git, GitHub |

## Machine Learning Models

The project compares the following classification algorithms:

| Model | Key Characteristics |
|---|---|
| Logistic Regression | Simple baseline model with low computational cost |
| Decision Tree | Interpretable model that may overfit |
| Random Forest | Ensemble model with improved stability |
| XGBoost | Gradient boosting with strong predictive performance |
| LightGBM | Efficient gradient boosting suitable for large structured datasets |
| Support Vector Machine (SVM) | Useful for certain classification problems but potentially challenging with imbalanced datasets |

### Selected Model: LightGBM

LightGBM is selected as the primary model in the proposed system for the following reasons:

- **Training efficiency:** Uses histogram-based learning to improve computational efficiency.
- **Scalability:** Supports large structured datasets.
- **Memory efficiency:** Uses optimized data structures.
- **Predictive performance:** Can model complex relationships between claim attributes and fraud outcomes.

Model selection should be based on evaluation against the same held-out test dataset. In fraud detection, recall, precision, F1-score, and ROC-AUC should be considered alongside accuracy because fraudulent claims may represent a small proportion of all claims.

## System Architecture

The application follows an end-to-end machine learning pipeline:

1. **Data Collection:** Load historical insurance claim records.
2. **Data Preprocessing:** Handle missing values, duplicates, categorical variables, and numerical features.
3. **Feature Engineering:** Transform relevant claim attributes into model-ready features.
4. **Class Imbalance Handling:** Apply SMOTE to the training data when appropriate.
5. **Model Training:** Train and compare classification algorithms.
6. **Model Evaluation:** Evaluate predictive performance using classification metrics.
7. **Model Persistence:** Save the selected trained model and preprocessing components.
8. **Real-Time Prediction:** Receive claim details through the Flask API and generate predictions.
9. **Risk Assessment:** Apply a configurable threshold to classify claims by estimated fraud risk.
10. **Dashboard and Feedback:** Display performance statistics, log predictions, and support future model improvement.

## Project Structure

The following is a suggested structure. Update the names to match the actual implementation.

```text
insurance-claims-fraud-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── fraud_detection_analysis.ipynb
│
├── models/
│   ├── fraud_model.joblib
│   └── preprocessor.joblib
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── api/
│   ├── app.py
│   ├── prediction_routes.py
│   └── dashboard_routes.py
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── README.md
│
├── logs/
│   └── predictions.csv
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset and Preprocessing

The model requires a labeled insurance claims dataset containing claim attributes and a target variable indicating whether a claim is fraudulent.

Typical preprocessing steps include:

- Removing duplicate records.
- Handling missing and inconsistent values.
- Encoding categorical features using OneHotEncoder or suitable alternatives.
- Scaling numerical features where required by the selected algorithm.
- Splitting the data into training, validation, and test sets.
- Handling class imbalance using SMOTE on training data only.
- Preventing data leakage by fitting preprocessing transformations exclusively on training data.

The exact feature names and target column depend on the dataset used.

## Model Evaluation

The system evaluates the trained model using the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified claims |
| Precision | Measures how many claims flagged as fraudulent are actually fraudulent |
| Recall | Measures how many known fraudulent claims are detected |
| F1-score | Balances precision and recall |
| ROC-AUC | Measures the model's ability to distinguish between the two classes |

For fraud detection, recall is especially important because missed fraudulent claims can create financial losses. Precision is also critical because excessive false positives increase investigation costs.

**Important:** Actual model metrics should be calculated from your test results. Do not treat illustrative or previously reported comparison values as verified production performance.

## Flask API

The backend exposes endpoints for fraud prediction and dashboard analytics.

### 1. Predict Insurance Claim Fraud

**Endpoint:** `POST /predict`

Purpose:
- Accept claim attributes in JSON format.
- Apply the saved preprocessing pipeline.
- Generate a fraud probability and classification.
- Determine the risk category using the configured threshold.
- Record the prediction for analysis.

Example request structure:

```json
{
  "claim_amount": 25000,
  "claim_duration_days": 15,
  "policy_age_days": 365
}
```

The fields above are illustrative. Replace them with the features expected by your actual trained model.

Example response structure:

```json
{
  "prediction": "Potential Fraud",
  "fraud_probability": 0.87,
  "risk_level": "High"
}
```

The response values are examples, not actual model output.

### 2. Configure Prediction Threshold

**Endpoint:** `POST /config/threshold`

Purpose:
- Update the fraud probability threshold.
- Adjust the trade-off between detecting more fraudulent claims and reducing false positives.

Example request:

```json
{
  "threshold": 0.65
}
```

The threshold must be between 0 and 1. It should be selected using validation data and business requirements.

### 3. Retrieve Dashboard Analytics

The dashboard API provides:

- Accuracy, precision, recall, F1-score, and ROC-AUC.
- Fraudulent versus genuine claim distribution.
- Feature importance rankings.
- Total predictions processed.
- Fraud and genuine prediction counts.

The exact route for this functionality should match the implemented Flask application.

## Installation and Setup

### Prerequisites

Install the following:

- Python 3.10 or a compatible version for the selected dependencies.
- Node.js and npm for the React frontend.
- Git.

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd insurance-claims-fraud-detection
```

Replace the placeholder with your GitHub repository URL.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

If you are creating the requirements file, include the libraries required by your implementation, such as:

```text
flask
pandas
numpy
scikit-learn
lightgbm
imbalanced-learn
joblib
```

Add any additional dependencies used by the project.

### Step 4: Prepare the Dataset

Place the dataset in the appropriate data directory and configure the training script to load the correct file and target column.

Ensure that sensitive personal or insurance information is not committed to a public repository.

### Step 5: Train the Model

If your training script is named `train.py`, the command may be:

```bash
python src/train.py
```

The script should preprocess the data, train the model, evaluate it, and save the model artifacts.

### Step 6: Start the Flask API

If the Flask application is defined in `api/app.py`, run:

```bash
python api/app.py
```

The API address and port depend on the application's configuration.

### Step 7: Start the Frontend

Open a separate terminal:

```bash
cd frontend
npm install
npm start
```

Use `npm run dev` instead if the frontend is configured with Vite.

## Dashboard Overview

The frontend dashboard can provide the following views:

- **Overview:** Total claims analyzed and the number flagged for review.
- **Model Performance:** Accuracy, precision, recall, F1-score, and ROC-AUC.
- **Claim Analysis:** Form for submitting claim details and viewing prediction results.
- **Risk Assessment:** Visual indicators for low-, medium-, and high-risk predictions.
- **Feature Importance:** Displays the attributes that contribute to the model's predictions.
- **Prediction History:** Shows previously recorded predictions and their outcomes where available.

Risk categories and dashboard statistics should be derived from the backend rather than hardcoded values.

## Security and Responsible Use

- Validate and sanitize all incoming API requests.
- Restrict access to claim data and prediction logs.
- Avoid exposing personally identifiable information in logs.
- Store secrets in environment variables rather than source code.
- Use authenticated access controls before deploying the system.
- Treat model predictions as investigation signals, not definitive proof of fraud.
- Require authorized human review before adverse claim decisions.
- Monitor false positives, false negatives, and model drift.

## Future Enhancements

- Integrate additional fraud indicators and historical claim patterns.
- Add explainability using SHAP or other feature attribution methods.
- Implement authenticated user access and role-based permissions.
- Deploy the application using Docker and a cloud hosting platform.
- Introduce model monitoring and periodic retraining.
- Add feedback from verified investigation outcomes to improve future models.
- Implement automated alerts for claims exceeding configured risk thresholds.

## Conclusion

The Insurance Claims Fraud Detection System combines machine learning, a Flask REST API, and interactive analytics to support intelligent insurance claim assessment.

By comparing classification algorithms, handling imbalanced datasets, evaluating fraud-specific metrics, and generating configurable risk scores, the system provides a foundation for more efficient claim investigation and data-driven fraud analysis.

## Author

**Rakesh Kumar V S**

B.Tech in Artificial Intelligence and Data Science

Sri Krishna College of Technology, Coimbatore

## License

Specify an appropriate license if the project will be distributed publicly. Use the MIT License only if you intend to release the project under those terms.
