import pandas as pd
import joblib
import os
    # Get the directory of the current file (e.g., scripts/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define a path relative to that file (e.g., ../models/)
MODEL_DIR = os.path.join(BASE_DIR, "..","..", "models")

class Utils:
    def read_csv(self,path):
        return pd.read_csv(path)
    def read_sql(self,databaseParams):
        pass
    def features_targets(self,dataset,drop_cols,target_cols):
        X=dataset.drop(drop_cols,axis=1)
        Y=dataset[target_cols]
        return X,Y
    def model_export (self,clf,score):
# Ensure the folder exists
        os.makedirs(MODEL_DIR, exist_ok=True)

# Save the model
        joblib.dump(clf, os.path.join(MODEL_DIR, "best_model.pkl"))

        print(score)
        print("Saving model in ", MODEL_DIR)
        print("Working directory:", os.getcwd())
    def model_load(self,name:str):
        model=os.path.abspath(os.path.join(MODEL_DIR,name))
        m=joblib.load(model)
        print('the model has been loaded from', model)
        return m