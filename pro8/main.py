
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Create FastAPI application
app = FastAPI(title="Iris Flower Prediction API")

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train Machine Learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

# Define input data format
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Iris Flower Prediction API is running"
    }

# Get available flower names
@app.get("/flowers")
def get_flowers():
    return {
        "flower_names": iris.target_names.tolist()
    }

# Prediction endpoint
@app.post("/predict")
def predict(data: IrisInput):
    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(input_data)
    flower_name = iris.target_names[prediction[0]]

    return {
        "prediction": flower_name
    }
