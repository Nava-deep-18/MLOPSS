from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
import joblib

# Create the API application
app = FastAPI(title="Iris Prediction API")

# Load the trained model only once when the app starts
model = joblib.load("model.joblib")

# Get the original Iris species names
iris = load_iris()
species_names = iris.target_names


# Define and validate the incoming input
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Home endpoint
@app.get("/")
def home():
    return {"message": "Iris Prediction API is running!"}


# Prediction endpoint
@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(model.predict(features)[0])

    return {
        "predicted_class": prediction,
        "predicted_species": str(species_names[prediction])
    }
