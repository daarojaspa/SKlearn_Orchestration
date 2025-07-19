from utils.utils import Utils
from models import Models
import os

if __name__=="__main__":
    utils=Utils()
    models=Models()

# Get absolute path to the folder where this script lives
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go UP one level (../)
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# Now point to the 'models/' folder
    DATA_DIR = os.path.join(PARENT_DIR, "data/third_part")

# Full path to model file
    CSV_PATH = os.path.join(DATA_DIR, "felicidad.csv")

# Optional: create folder and save
    #os.makedirs(DATA_DIR, exist_ok=True)

    print("data will be read from", DATA_DIR)
    data=utils.read_csv(CSV_PATH)
    X,Y=utils.features_targets(data,["score","country","rank"],["score"])
    models.grid_trinning(X,Y)
    print("Done")