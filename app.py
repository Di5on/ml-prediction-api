import joblib
import pandas as pd
import logging
from fastapi import FastAPI
from pydantic import BaseModel,Field

logging.basicConfig(level=logging.INFO)

model = joblib.load("model/game_sales_model.pkl")
model_v2 = joblib.load("model/game_sales_model_v2.pkl")

app = FastAPI(
    title="Video Game Sales Prediction API",
    description="Predicts global video game sales using platform, genre, publisher, and regional sales.",
    version="1.0"
)

class GameInput(BaseModel):
    Platform: str
    Genre: str
    Publisher: str
    NA_Sales: float = Field(ge=0)
    EU_Sales: float = Field(ge=0)
    JP_Sales: float = Field(ge=0)
    Other_Sales: float = Field(ge=0)
@app.get("/")

def home():
    return {"message": "Video Game Sales Prediction API is running"}

@app.post("/predict")
def predict_sales(game: GameInput):
    logging.info(f"Received input: {game}")
    try:
        input_data = pd.DataFrame([game.dict()])
        prediction = model.predict(input_data)[0]

        return {
                "status": "success",
                "prediction": {
                    "global_sales": round(float(prediction), 2)
                }
            }

    except Exception as e:
        return {
            "error": str(e)
        }



class GameInputV2(BaseModel):
    Platform: str
    Genre: str
    Publisher: str
    Year: float

@app.post("/predict_v2")
def predict_sales_v2(game: GameInputV2):
    logging.info(f"Received input: {game}")
    try:
        input_data = pd.DataFrame([game.dict()])
        prediction = model_v2.predict(input_data)[0]

        return {
            "model": "v2",
            "predicted_global_sales": round(float(prediction), 2)
        }
    
    except Exception as e:
        return {
            "error": str(e)
        }