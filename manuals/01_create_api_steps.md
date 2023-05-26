# API - creation and deployment

- [API - creation and deployment](#api---creation-and-deployment)
  - [API setup](#api-setup)
  - [How to run API?](#how-to-run-api)
  - [API deployment](#api-deployment)
  - [API creation](#api-creation)
  - [How to add a new model?](#how-to-add-a-new-model)
- [Steps to create API](#steps-to-create-api)
  - [Info](#info)
  - [FastAPI](#fastapi)
  - [Steps](#steps)
  - [Using pretrained models](#using-pretrained-models)
  - [Steps to run API](#steps-to-run-api)
  - [Pretrained models from HuggingFace](#pretrained-models-from-huggingface)

The API was inspired by the 

## API setup

Install all project requirements with `pip`:

```bash
pip install -r requirements.txt
```

## How to run API?

We'll be using Uvicorn, a fast ASGI server (it can run asynchronous code in a single process) to launch our application. Use the following command to start the server:

```bash
uvicorn app.api:app \
    --host 0.0.0.0 \
    --port 5000 \
    --reload \
    --reload-dir app \
    --reload-dir models
```
Or
```
uvicorn app.api:app  --host 0.0.0.0 --port 8000  --reload  --reload-dir deploy-GAISSA --reload-dir app 
```
In detail:

- `uvicorn app.api:app` is the location of app (`app` directory > `api.py` script > `app` object);
- `--reload` makes the server reload every time we update;
- `--reload-dir app` makes it only reload on updates to the `app/` directory;
- `--reload-dir models` makes it also reload on updates to the `models/` directory;

**Observation**. If we want to manage multiple `uvicorn` workers to enable parallelism in our application, we can use **Gunicorn** in conjunction with **Uvicorn**.

<center><figure>
  <img
  src="images/01_api_running.png"
  <figcaption>API running.</figcaption>
</figure></center>

Now you can test the app:

We can now test that the application is working. These are some of the possibilities:

- Visit [localhost:5000](http://localhost:5000/)
- Use `curl`

  ```bash
  curl -X GET http://localhost:5000/
  ```

- Access the API programmatically, e.g.:

  ```python
  import json
  import requests

  response = requests.get("http://localhost:5000/")
  print (json.loads(response.text))
  ```

- Use an external tool like [Postman](https://www.postman.com), which lets you execute and manage tests that can be saved and shared with others.

Visit [localhost:5000/docs](http://localhost:5000/docs) and select one of the models.

<center><figure>
  <img
  src="images/01_api_ui.png"
  <figcaption>API User Interface in localhost:5000/docs endpoint.</figcaption>
</figure></center>

To make an inference, click on "Try it out" button and click execute.

You should obtain a "200" code response after executing the POST method of the model:

<center><figure>
  <img
  src="images/01_api_response_ui.png"
  <figcaption>API response on UI.</figcaption>
</figure></center>

<center><figure>
  <img
  src="images/01_api_response_terminal.png"
  <figcaption>API response on terminal.</figcaption>
</figure></center>

## API deployment
...

## API creation
The API in this project is freely inspired by the [Made with ML](https://madewithml.com) tutorial: "[APIs for Machine Learning](https://madewithml.com/courses/mlops/api/)" and [FastAPI Lab](https://github.com/se4ai2122-cs-uniba/SE4AI2021Course_FastAPI-demo).

Following the guide https://madewithml.com/courses/mlops/api/

## How to add a new model?

1. Add Models_names
2. Add ML_task
3. Create new class:
  def class NewModel(Model):
1. Create schema in schemas
2. Add endpoint in api


# Steps to create API


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
        You only have the weights 


## Steps to run API
1. Run server
   1. uvicorn app.api:app  --host 0.0.0.0 --port 8000  --reload  --reload-dir deploy-GAISSA --reload-dir app   
2. go to http://127.0.0.1:8000/docs
3. Use huggingface_model


## Pretrained models from HuggingFace

- What can be imported in old cpu
  - from transformers import TFBertTokenizer
  - 