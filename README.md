# Video Game Sales Prediction API

This project is an end-to-end machine learning API that predicts global video game sales using regional sales and game information.

---

## Tech Stack

* Python
* Pandas
* Scikit-learn
* FastAPI
* Uvicorn
* Joblib

---

## Features

* Trains a machine learning regression model
* Saves the trained model
* Provides REST API for real-time predictions
* Includes automatic API documentation using FastAPI Swagger UI

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run the following command to train the machine learning model:

```bash
python train_model.py
```

---

## Model Details

* Model: RandomForestRegressor
* Features:

  * Platform, Genre, Publisher
  * Regional sales (NA, EU, JP, Other)
* Target:

  * Global_Sales
* Metrics:

  * Mean Absolute Error (MAE)
  * R² Score

---

## Run the API

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Once the API is running, go to:

```
http://127.0.0.1:8000/docs
```

This will open the Swagger UI to test the API.

---

## Example Request

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
├── requirements.txt
├── README.md
├── data/
├── model/
```
