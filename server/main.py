import uvicorn
import io 
import h2o
import os
import pandas as pd

from fastapi import FastAPI, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# Set the experiment and run IDs
exp_id = '465920473530289427'
run_id = '5dc1ce13bb3844f4b3c68059f8f4b5ba'

if os.path.exists(f"mlruns/{exp_id}/{run_id}/artifacts/model/") and os.path.isdir(f"mlruns/{exp_id}/{run_id}/artifacts/model/"):
    print("The directory exists.")
else:
    print("The directory does not exist.")

app = FastAPI(   
    title='Cross Sell Classifier API',
    description='Predict if a current health policy customer will be interested in vehicle insurance bundling')
h2o.init()
client = MlflowClient()

# todo generate h20 model from directory
best_model = mlflow.h2o.load_model(f"mlruns/{exp_id}/{run_id}/artifacts/model/")
print('[+] Model Loaded')

@app.get("/", tags=["root"])
def root():
    """
    Root endpoint of the API.
    Returns a simple HTML response.
    """
    content = """
    <body>
    <h2> MSDS 498 Project - Insurance cross selling classification
    </body>
    """

    return HTMLResponse(content=content)

@app.post("/predict", tags=["predict"])
async def predict(file: bytes = File(...)):
    """
    Predict endpoint of the API.
    Accepts a CSV file as input and returns the predictions made by the best model.

    Parameters:
    - file: bytes, required. The CSV file containing the data to be predicted.

    Returns:
    - JSONResponse: The predictions made by the best model in JSON format.
    """
    print('[+] Initiate Prediction')
    file_obj = io.BytesIO(file)
    test_df = pd.read_csv(file_obj)
    test_h2o = h2o.H2OFrame(test_df)

    # Generate predictions with best model (output is H2O frame)
    preds = best_model.predict(test_h2o)
    print('[+] Prediction Completed')
    preds_final = preds.as_data_frame()['predict'].tolist()

    json_compatible_item_data = jsonable_encoder(preds_final)
    return JSONResponse(content = json_compatible_item_data)



