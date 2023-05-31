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

Guide:
1. [API creation](manuals/01_create_api.md). Guide to create an API to deploy ML models.
2. [Add pretrained model](manuals/02_add_models.md). (ToDo) Guide to add pretrained ML models (from HuggingFace, hdf5 format, pickle format) to do inferences through an API.
3. [Deploy ML models in a cloud provider (General)](manuals/03_deploy_general.md). Guide to deploy ML models using an API in a cloud provider.
4. [Deploy in Virtech](manuals/04_deploy_fib.md). Virtech setup, Guide to deploy ML models using an API in an AWS VM.
5. [AWS](manuals/05_deploy_aws.md). AWS setup, Guide to deploy ML models using an API in an AWS VM.
7. [GCP](manuals/06_deploy_gcp.md). GCP setup, Guide to deploy ML models using an API in an GCP VM.
8. [Azure](manuals/). (ToDo) Azure setup, Guide to deploy ML models using an API in an Azure VM.
9. [FAQ](manuals/FAQ.md). (ToDo) Documentation with problems arised during deployments.


## Cloud providers*

\* Initial proposed cloud providers

<pre/>
- Amazon Elastic Compute Cloud (Amazon EC2) from Amazon Web Services (AWS)
  | URL: https://aws.amazon.com/
- Azure Virtual Machines from Microsoft Windows Azure
  | URL: https://azure.microsoft.com/
- Google Compute Engine from Google Cloud Platform (GCP)
  | URL: https://cloud.google.com/
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