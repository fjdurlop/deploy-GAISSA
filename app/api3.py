# app/api.py

from datetime import datetime
from functools import wraps
from fastapi import FastAPI, Request

from http import HTTPStatus
from typing import Dict

from pathlib import Path
#from config import config
#from logging.config import logger
#from tagifai import main


# To run the API
# uvicorn app.api:app  --host 0.0.0.0 --port 8000  --reload  --reload-dir deploy-GAISSA --reload-dir app 


# Define application
app = FastAPI(
    title="deploy-GAISSA",
    description="ML app",
    version="0.1",
)

def construct_response(f):
    """Construct a JSON response for an endpoint."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


# Load artifacts for the model to inference
# Service won't accept requests until this is complete
# @app.on_event("startup")
# def load_artifacts():
#     global artifacts
#     run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
#     artifacts = main.load_artifacts(model_dir=config.MODEL_DIR)
#     logger.info("Ready for inference!")
    

@app.get("/", tags=["General"]) # defines the path for the endpoint as well as other attributes
@construct_response
def _index(request:Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response

@app.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: str = None) -> Dict:
    """Get the performance metrics."""
    performance = artifacts["performance"]
    data = {"performance":performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response