import joblib 
import numpy as np
from utils.utils import Utils
from flask import Flask, jsonify
app =Flask(__name__)
@app.route('/predict',methods=["GET"])
def predict():
        x=[5.897366465,5.722633421,1.346911311,1.186303377,0.834647238,0.471203625,0.266845703,0.155353352,1.549157619]
        X_test=np.array(x)
        prediction=model.predict(X_test.reshape(1,-1))
        return jsonify({'prediction':list(prediction)})

    
if __name__=="__main__":
    utils=Utils()
    model=utils.model_load("best_model.pkl")
    if model is None:
        raise ValueError("Model loaded as None. Something went wrong.")

    if not hasattr(model, "predict"):
        raise TypeError("Loaded object is not a valid model. No .predict() method.")

    print(type(model))
    app.run(port=12500)