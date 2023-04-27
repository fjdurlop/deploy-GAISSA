# Steps to create API
Following the guide https://madewithml.com/courses/mlops/api/

## Info
In this API, the client sends a request with the appropiate inputs to the server (Application with a trained model) and receives a response.

cURL to execute the API calls.
curl -X GET "http://localhost:8000/models"

## FastAPI
Framework to build API service.
Other options: Flask, Django and even non-Python based options like Node, Angular, etc.

## Steps
1. Set environment
   1. app directory
      1. api.py - FastAPI app
         1. Swagger UI automatically created
      2. gunicorn.py - WSGI script
      3. schemas.py - API model schemas


uvicorn app.api:app  --host 0.0.0.0 --port 8000  --reload  --reload-dir deploy-GAISSA --reload-dir app        





## Using pretrained models
https://discuss.huggingface.co/t/how-to-save-my-model-to-use-it-later/20568

Errors
    if No model found in config file.
        Yo only have the weights 