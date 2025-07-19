import joblib
import numpy as np
model = joblib.load("best_model.pkl")
print("Type:", type(model))             # Should show a scikit-learn class
print("Has predict:", hasattr(model, "predict"))  # Should be True
print(model)  # View model details and parameters
x=[5.897366465,5.722633421,1.346911311,1.186303377,0.834647238,0.471203625,0.266845703,0.155353352,1.549157619]
X_test=np.array(x)
preds = model.predict(X_test.reshape(1,-1))
        

print(preds[:5])