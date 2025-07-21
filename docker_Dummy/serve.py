import pickle
from fastapi import FastAPI
import uvicorn

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def read_root():
    return {"message": "Model is ready"}

@app.get("/predict/")
def predict():
    return {"prediction": model.predict([[5.1, 3.5, 1.4, 0.2]]).tolist()}
