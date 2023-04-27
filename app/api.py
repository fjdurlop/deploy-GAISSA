"""Main script: it includes our API initialization and endpoints."""

import pickle
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, Request

from app.schemas import IrisType, PredictPayload, PredictBert

from transformers import pipeline


MODELS_DIR = Path("models/")
NAME_APP = "deploy-GAISSA"
model_wrappers_list: List[dict] = []

# Define application
app = FastAPI(
    title=NAME_APP,
    description="This API lets you make predictions on .. using a couple of simple models.",
    version="0.1",
)


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


@app.on_event("startup")
def _load_models():
    """Loads all pickled models found in `MODELS_DIR` and adds them to `models_list`"""

    model_paths = [
        filename for filename in MODELS_DIR.iterdir() if filename.suffix == ".pkl"
    ]

    for path in model_paths:
        with open(path, "rb") as file:
            model_wrapper = pickle.load(file)
            model_wrappers_list.append(model_wrapper)

    

@app.get("/", tags=["General"])  # path operation decorator
@construct_response
def _index(request: Request):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": f"Welcome to {NAME_APP}! Please, read the `/docs`!"},
    }
    return response


@app.get("/models", tags=["Pickle Models"])
@construct_response
def _get_models_list(request: Request):
    """Return the lsit of available models"""

    available_models = [
        {
            "type": model["type"],
            "parameters": model["params"],
            "accuracy": model["metrics"],
        }
        for model in model_wrappers_list
    ]

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": available_models,
    }

    return response


@app.post("/models/{type}", tags=["Pickle Models"])
@construct_response
def _predict(request: Request, type: str, payload: PredictPayload):
    """Classifies Iris flowers based on sepal and petal sizes."""

    # sklearn's `predict()` methods expect a 2D array of shape [n_samples, n_features]
    # therefore, we need to convert our single data point into a 2D array
    features = [
        [
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width,
        ]
    ]

    model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    if model_wrapper:

        prediction = model_wrapper["model"].predict(features)
        prediction = int(prediction[0])
        predicted_type = IrisType(prediction).name

        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                "model-type": model_wrapper["type"],
                "features": {
                    "sepal_length": payload.sepal_length,
                    "sepal_width": payload.sepal_width,
                    "petal_length": payload.petal_length,
                    "petal_width": payload.petal_width,
                },
                "prediction": prediction,
                "predicted_type": predicted_type,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response

@app.post("/huggingface_models/bert", tags=["Hugging Face Models"])
@construct_response
def _predict_bert(request: Request, payload: PredictBert):
    """bert-base-uncased model."""

    # sklearn's `predict()` methods expect a 2D array of shape [n_samples, n_features]
    # therefore, we need to convert our single data point into a 2D array
    # features = [
    #     [
    #         payload.sepal_length,
    #         payload.sepal_width,
    #         payload.petal_length,
    #         payload.petal_width,
    #     ]
    # ]
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    #model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    if input_text:

        # prediction = model_wrapper["model"].predict(features)
        # prediction = int(prediction[0])
        # predicted_type = IrisType(prediction).name

        unmasker = pipeline('fill-mask', model='./bert-model')
        output = unmasker(input_text)
        print(output)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                #"model-type": model_wrapper["type"],
                "model-type": "BERT",
                "input_text": input_text,
                "prediction": output,
                #"predicted_type": predicted_type,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response