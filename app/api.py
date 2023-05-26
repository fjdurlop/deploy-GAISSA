"""Main script: it includes our API initialization and endpoints.

Problems 

>>> from transformers import pipeline
>>> import tensorflow
Illegal instruction (core dumped)

In brief, the error will be thrown if we’re running recent TensorFlow binaries on CPU(s) 
that do not support Advanced Vector Extensions (AVX), an instruction set that enables faster
computation especially for vector operations. Starting from TensorFlow 1.6, pre-built
TensorFlow binaries use AVX instructions. An excerpt from TensorFlow 1.6 release announcement: 
tf 1.6 - feb 18
transformers - 19
https://tech.amikelive.com/node-887/how-to-resolve-error-illegal-instruction-core-dumped-when-running-import-tensorflow-in-a-python-program/

flags           : fpu de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pse36 clflush mmx fxsr sse sse2 syscall nx lm rep_good nopl xtopology cpuid tsc_known_freq pni cx16 x2apic hypervisor lahf_lm cpuid_fault pti

CPU features
    windows
        check system information, then search {cpu model} CPU features
    linux
        more /proc/cpuinfo | grep flags
        
This repository is tested on Python 3.6+, Flax 0.3.2+, PyTorch 1.3.1+ and TensorFlow 2.3+.


"""

import pickle
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, Request

from app.schemas import IrisType, PredictPayload, PredictBert, PredictT5, PredictCNN, PredictCodeGen, PredictPythia_70m, PredictCodet5p_220m

#from transformers import pipeline

# Local modules
from app.models import LMBERTModel, Model, T5Model, CNNModel, CodeGenModel, Pythia_70mModel, Codet5p_220mModel

print("------------------------modules loaded!------------------------")

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
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    #model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    model = LMBERTModel()
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                #"model-type": model_wrapper["type"],
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
                #"predicted_type": predicted_type,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response


@app.post("/huggingface_models/t5", tags=["Hugging Face Models"])
@construct_response
def _predict_t5(request: Request, payload: PredictT5):
    """T5 model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    #model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    model = T5Model()
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                #"model-type": model_wrapper["type"],
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
                #"predicted_type": predicted_type,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response


@app.post("/huggingface_models/CodeGen", tags=["Hugging Face Models"])
@construct_response
def _predict_codegen(request: Request, payload: PredictCodeGen):
    """CodeGen model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    #model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    model = CodeGenModel()
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                #"model-type": model_wrapper["type"],
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
                #"predicted_type": predicted_type,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response



@app.post("/huggingface_models/Pythia_70m", tags=["Hugging Face Models"])
@construct_response
def _predict_pythia_70m(request: Request, payload: PredictPythia_70m):
    """T5 model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    #model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    model = Pythia_70mModel()
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                #"model-type": model_wrapper["type"],
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
                #"predicted_type": predicted_type,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response

CodeGenModel, Pythia_70mModel, Codet5p_220mModel

@app.post("/huggingface_models/Codet5p_220m", tags=["Hugging Face Models"])
@construct_response
def _predict_codet5p_220m(request: Request, payload: PredictCodet5p_220m):
    """Codet5p_220m model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    #model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    model = Codet5p_220mModel()
    print(f"Model: {model.name}")

    if input_text:
        prediction = model.predict(input_text)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                #"model-type": model_wrapper["type"],
                "model-type": model.name,
                "input_text": input_text,
                "prediction": prediction,
                #"predicted_type": predicted_type,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response


@app.post("/h5_models/cnn_fashion", tags=["H5 Models"])
@construct_response
def _predict_cnn(request: Request, payload: PredictCNN):
    """CNN model."""
    
    input_text = payload.input_text 
    print("Input text")
    print(input_text)
    #model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    model = CNNModel()
    print(f"Model: {model.name}")

    if input_text:
        #prediction = model.predict(image)
        model_response = model.predict(input_text)
        
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                #"model-type": model_wrapper["type"],
                "model-type": model.name,
                "input_text": input_text,
                "prediction": model_response['prediction'],
                "is_correct": model_response['is_correct'],
                #"predicted_type": predicted_type,
            },
        }
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response

# @app.post("/huggingface_models/bert", tags=["Hugging Face Models"])
# @construct_response
# def _predict_bert(request: Request, payload: PredictBert):
#     """bert-base-uncased model."""

#     # sklearn's `predict()` methods expect a 2D array of shape [n_samples, n_features]
#     # therefore, we need to convert our single data point into a 2D array
#     # features = [
#     #     [
#     #         payload.sepal_length,
#     #         payload.sepal_width,
#     #         payload.petal_length,
#     #         payload.petal_width,
#     #     ]
#     # ]
    
#     input_text = payload.input_text 
#     print("Input text")
#     print(input_text)
#     #model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

#     if input_text:

#         # prediction = model_wrapper["model"].predict(features)
#         # prediction = int(prediction[0])
#         # predicted_type = IrisType(prediction).name

#         unmasker = pipeline('fill-mask', model='./bert-base-uncased')
#         output = unmasker(input_text)
#         print(output)
        
#         response = {
#             "message": HTTPStatus.OK.phrase,
#             "status-code": HTTPStatus.OK,
#             "data": {
#                 #"model-type": model_wrapper["type"],
#                 "model-type": "BERT",
#                 "input_text": input_text,
#                 "prediction": output,
#                 #"predicted_type": predicted_type,
#             },
#         }
#     else:
#         response = {
#             "message": "Model not found",
#             "status-code": HTTPStatus.BAD_REQUEST,
#         }
#     return response