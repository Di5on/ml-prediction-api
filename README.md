# Video Game Sales Prediction API

This project is an end-to-end machine learning system that predicts global video game sales using a trained regression model deployed via a FastAPI REST API.

---

## Overview

The project demonstrates a complete ML workflow:

* Data preprocessing and feature selection
* Model training and evaluation
* Model version comparison
* Deployment using FastAPI for real-time predictions

---

## Tech Stack

* Python
* Pandas
* Scikit-learn
* FastAPI
* Uvicorn
* Joblib

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Models

### Version 1 (Baseline Model)

```bash
python train_model.py
```

### Version 2 (Improved Model)

```bash
python train_model_v2.py
```

---

## Model Versions

### Model V1 (Baseline)

* **Features:**

  * Platform, Genre, Publisher
  * NA_Sales, EU_Sales, JP_Sales, Other_Sales
* **Target:** Global_Sales
* **Description:**

  * Uses regional sales to predict global sales
  * Achieves higher accuracy
  * Easier prediction problem

---

### Model V2 (Improved / Realistic)

* **Features:**

  * Platform, Genre, Publisher, Year
* **Target:** Global_Sales
* **Description:**

  * Removes regional sales to avoid data leakage
  * More realistic prediction scenario
  * Lower R² score (~ -0.01)

---

## Model Comparison

| Model | Features Used           | Result        | Purpose                           |
| ----- | ----------------------- | ------------- | --------------------------------- |
| V1    | Includes regional sales | High accuracy | Demonstrates deployment           |
| V2    | Excludes regional sales | Low accuracy  | Demonstrates realistic ML problem |

### Key Insight

Model V1 performs well because regional sales are strongly correlated with global sales, making the prediction easier.

Model V2 removes these features to avoid data leakage. The lower performance highlights that features like platform, genre, publisher, and year alone are not strong predictors of global sales.

This demonstrates the importance of feature selection in machine learning.

---

## Run the API

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Then open:

```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Predict (Model V1)

```
POST /predict
```

### Predict (Model V2)

```
POST /predict_v2
```

---

## Example Request (V1)

```json
{
  "Platform": "PS2",
  "Genre": "Sports",
  "Publisher": "Electronic Arts",
  "NA_Sales": 1.5,
  "EU_Sales": 1.2,
  "JP_Sales": 0.2,
  "Other_Sales": 0.5
}
```

---

## Example Response

```json
{
  "predicted_global_sales": 3.27
}
```

---

## Project Structure

```
ml-prediction-api/
├── app.py
├── train_model.py
├── train_model_v2.py
├── requirements.txt
├── README.md
├── data/
│   └── vgsales.csv
├── model/
│   ├── game_sales_model.pkl
│   └── game_sales_model_v2.pkl
```

---

## Key Features

* End-to-end ML pipeline
* REST API for real-time predictions
* Multiple model versions for comparison
* Demonstrates data leakage vs realistic modeling

---