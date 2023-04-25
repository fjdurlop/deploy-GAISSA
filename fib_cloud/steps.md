## Steps 


## Deploying in VM
1. Get VM
   1. It should have more than 8 gb of disk space
2. Get ip addr
   1. 10.4.41.62
3. Optional – win – connect from powershell
   1. ssh alumne@10.4.41.62
4. Check memory 
5. From vm 
   1. Free -h : 2.4 Gi  , RAM memory
   2. df -h, disk space
   3. From model
   4. When testing model, use htop or top to check the memory it uses
6. Installing software
   1. Sudo apt update / Update your local system's repository list by entering the following command
   2. Sudo apt upgrade
   3. Install Python
   4. Install pip
   5. Install modules
      1. Transformers
      2. torch or tf
7. Using pretrained model
   1. https://huggingface.co/bert-base-uncased
```
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

# To use it
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
unmasker("Hello I'm a [MASK] model.")
```
8. Other
Explain me how to use a pretrained Huggingface ml model from a VM

## Software and modules

### Install
- python3
- pip 
### Modules
- transformers
- torch or tf
- requirements: https://github.com/se4ai2122-cs-uniba/SE4AI2021Course_FastAPI-demo

Disk space used: 9.1G

## How to use pretrained Huggingface model 

1. Install Hugging Face "transformers" module
2. Load pre-trained model
3. Load pretrained model
```
from transformers import AutoModel

model = AutoModel.from_pretrained('bert-base-uncased')
```
4. Tokenize the input text:
```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
input_text = "Hello, world!"
tokenized_input = tokenizer(input_text, return_tensors='pt')

```
5. Run the model
```
# Run model
outputs = model(**tokenized_input)
```

## how can I open a swagger UI that is in a remote server, in which I can connect by ssh 
- run server in remote machine
Create SSH tunnel to access endpoints, forwards traffic from my local port XXXX to the remote server's port XXXX
- It has to be running

```
ssh -L 8080:localhost:8080 myusername@123.45.67.89
ssh -L 5000:localhost:5000  alumne@10.4.41.62
```