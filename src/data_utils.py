import os
import pandas as pd

def load_synthetic_data():
    file = "data/synthetic_crime_scenes.csv"

    if not os.path.exists(file):
        print("No such file")
        return None
    
    df = pd.read_csv(file)
    df = df.dropna()

    return df