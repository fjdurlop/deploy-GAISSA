# deploy-GAISSA
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Summary
Guidelines to deploy AI models in different cloud providers aligned with green AI goals.

## Repository Structure

The repository is structured as follows:

<pre/>
- app
  | API, schemas
- models
  | This folder contains our trained or pretrained models
- notebooks
  | This folder contains the jupyter notebooks
- reports
  | Generated PDFs, graphics and figures to be used in reporting
- utils
  | Python functions
- manuals
  | self-contained manuals
- requirements.txt: The dependencies of our implementation
</pre>

Guide (A self-contained manual for each task):
1. [API creation](manuals/01_create_api_steps.md). See manuals/api to check how to create an API.
2. [Adding model](manuals/). See --- to check how to add a pretrained model from hugging face, h5 ... into an API and inference
3. [AWS](manuals/). AWS setup, how to deploy a model in an AWS vm
4. [Azure](manuals/). Azure setup, how to deploy a model in an Azure vm
5. [GCP](manuals/). GCP setup, how to deploy a model in an GCP vm
6. [Virtech](manuals/). Virtech setup,how to deploy a model in an Virtech vm

## Cloud providers*

\* Initial proposed cloud providers

<pre/>
- Amazon Elastic Compute Cloud (Amazon EC2) from Amazon Web Services (AWS)
  | URL: https://aws.amazon.com/
- Azure Virtual Machines from Microsoft Windows Azure
  | URL: https://azure.microsoft.com/
- Google Compute Engine from Google Cloud Platform (GCP)
  | URL: http://www.cs.toronto.edu/~kriz/cifar.html
- Virtech, UPC cloud provider (By OpenNebula)
  | URL: https://www.fib.upc.edu/es/la-fib/servicios-tic/cloud-docente-fib
  | URL: https://opennebula.io/
</pre>

### Amazon EC2
### Azure Virtual Machines
### Google Compute Engine
### Virtech, UPC cloud provider


## Models*
\* Initial proposed models

### Text Generation
- BERT
  - https://huggingface.co/bert-base-uncased

- T5
  - https://huggingface.co/t5-base
### Computer Vision
- CNN model
  - https://github.com/fjdurlop/guided-retraining/tree/main/models

### Code Generation
- CodeGen
  - https://huggingface.co/Salesforce/codegen-350M-mono
- Pythia-70m
  - https://huggingface.co/EleutherAI/pythia-70m
- Codet5p-220m
  - https://huggingface.co/Salesforce/codet5p-220m

## API
see manuals/01_create_api

### FastAPI

## ML frameworks
- TensorFlow
- PyTorch

## ML model formats
- ONNX
- h5, complete model
- h5, weights only
- Pickle

## ML task
- CV
- NLP
- ...

## Roles
Role: ML Engineer

- Data engineer: Manage DBs
- Data scientist: Train ML models
- BI: Dashboards, analytics, BI
- ML Engineer: SE --deploy--> ML systems

## Energy tracking variables
- codecarbon
- ...

## Future work
- Track energy efficiency.
- Trade-off between green-AI related metrics and accuracy.
- Monitor models' performance

## References
See manuals/references