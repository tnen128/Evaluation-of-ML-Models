# Credit Card Approval Prediction

This project predicts credit card approval outcomes based on various applicant features using machine learning. It involves data analysis, model training, model interpretation, and a Flask API to serve predictions from a logistic regression model.

## Table of Contents
- Project Overview
- Data Analysis and Model Training
- Interpretation of Results
- Flask API for Predictions
- Testing the API
- Example Input and Output

---

## Project Overview

This project builds and interprets a model to predict credit card approval outcomes using several applicant features:
- **Num_Children**: Number of children the applicant has
- **Income**: Annual income of the applicant
- **Own_Car**: Whether the applicant owns a car
- **Own_Housing**: Whether the applicant owns housing
- **Gender**: Gender of the applicant (Male or Female)

The analysis includes logistic regression and XGBoost models and assesses model interpretability, fairness, and bias metrics.

## Data Analysis and Model Training

- **Exploratory Data Analysis (EDA)**: Examined distributions, checked for class imbalances, and inspected relationships between features.
- **Feature Engineering**: Applied encoding for categorical variables and standardization for numeric variables.
- **Model Training**: Trained logistic regression and XGBoost models, evaluated using accuracy, precision, recall, F1 score, and Mean Squared Error (MSE).
- **Fairness & Bias Metrics**: Analyzed statistical parity, true positive rate (TPR), and false positive rate (FPR) across groups.
- **SHAP Analysis**: Used SHAP values to interpret feature importance and their effects on predictions.

## Interpretation of Results

- **Logistic Regression**:
  - High accuracy and balanced precision/recall.
  - Minimal overfitting observed from training and test scores.
  - Income and gender were identified as significant features impacting approval likelihood.
- **XGBoost**:
  - Comparable performance to logistic regression, with slightly lower accuracy and recall.
  - Similar feature importance ranking as logistic regression.
  - Slightly better fairness for Gender.

Overall, both models performed well, with logistic regression providing slightly better performance than XGBoost.

## Flask API for Predictions

This project includes a Flask API that uses the logistic regression model for predictions. The API takes JSON-formatted input and returns credit card approval predictions.


Install the required packages using:
```bash
pip install -r requirements.txt
```

### Running the API

1. Clone the repository:

2. Start the Flask server:
```bash
python app.py
```

By default, the Flask server will run on `http://127.0.0.1:5000/`.

- Endpoint
```
http://127.0.0.1:5000/predict
```

### Input Format

The JSON input should be like this

```json
{
  "Num_Children": [1, 1],
  "Gender": ["Male", "Male"],
  "Income": [40690, 70497],
  "Own_Car": ["No", "Yes"],
  "Own_Housing": ["Yes", "Yes"]
}

```

- **Num_Children**
- **Gender**
- **Income**
- **Own_Car**
- **Own_Housing**


### Response Format

A successful response will look like this
```json
{
    "predictions": [
        0,
        1
    ]
}
```
- **Approved_(1)**
- **Denied_(0)**

### Error Handling

- Missing Required Field: If any required field is missing, the API returns an error message like:

```json
{
    "error": "Missing required field: Own_Car"
}

```

- Inconsistent List Lengths: If the lists in input data are not of equal length, the API returns:
- 
```json

{
    "error": "All input lists must be of the same length."
}

```

### Testing the API Using Postman

1. Open Postman.
2. Create a new POST request.
3. Enter the following URL: `http://127.0.0.1:5000/predict`.
4. In the "Body" tab, select "raw" and choose "JSON" as the format.
5. Paste your input data as shown above.
6. Click "Send" to receive the prediction.


