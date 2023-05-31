**ToDo:**

- [ ] Add all problems in this directory
- [ ] Organize
- [ ] Update if new problems

-------------------
- [FAQ and problems arised during deployment](#faq-and-problems-arised-during-deployment)
  - [problems](#problems)
- [torch tensors](#torch-tensors)
  - [Moving to CPU, GPU](#moving-to-cpu-gpu)
    - [Devices](#devices)
  - [problems](#problems-1)
  - [Deploying in VM Virtech](#deploying-in-vm-virtech)
  - [How to use pretrained Huggingface model](#how-to-use-pretrained-huggingface-model)
  - [how can I open a swagger UI that is in a remote server, in which I can connect by ssh](#how-can-i-open-a-swagger-ui-that-is-in-a-remote-server-in-which-i-can-connect-by-ssh)
  - [Steps](#steps)
  - [Errors in Virtech](#errors-in-virtech)

# FAQ and problems arised during deployment
--------------------

## problems
torch1.* has the problem with  low_cpu_mem_usage=True

when ading low_cpu_mem_usage=True
    RuntimeError: Tensor on device meta is not on the expected device cpu!

https://huggingface.co/docs/transformers/main_classes/model

check if I can obtain model with device_map="auto", which reduces the memory usage
- Bert model is not 
- ValueError: BertForMaskedLM does not support `device_map='auto'` yet
model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased', return_dict = True, device_map="auto")

t5, _no_split_modules
https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L785

bert
https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
does not have that attribute

condition that prevents bert model of using device_map
https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2691


By passing `device_map="auto"`, we tell Accelerate to determine automatically where to put each layer of the model depending on the available resources:

no_split_module_classes (`List[str]`):
A list of class names for layers we don't want to be split.
https://github.com/huggingface/transformers/issues/23086


--------------------

- Trying to use pipeline with old or small CPU/GPU: Illegal instruction (core dumped)
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

--------------------

# torch tensors

Create torch tensor from python data

```
some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)
```


This does not copy, it is just a label
```
a = torch.ones(2, 2)
b = a
```

To copy:
```
a = torch.ones(2, 2)
b = a.clone()
```

## Moving to CPU, GPU

- To do computing tensors must be in the same device

```
if torch.cuda.is_available():
    print('We have a GPU!')
else:
    print('Sorry, CPU only.')
```


```
device = torch.device("cpu")

```

```
if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

x = torch.rand(2, 2, device=my_device)
print(x)
```


Moving to another device
```
y = torch.rand(2, 2)
y = y.to(my_device)
```
### Devices
- meta device: Tensor without any data attached to it

--------------------


## problems
torch1.* has the problem with  low_cpu_mem_usage=True

when ading low_cpu_mem_usage=True
    RuntimeError: Tensor on device meta is not on the expected device cpu!

https://huggingface.co/docs/transformers/main_classes/model

check if I can obtain model with device_map="auto", which reduces the memory usage
- Bert model is not 
- ValueError: BertForMaskedLM does not support `device_map='auto'` yet
model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased', return_dict = True, device_map="auto")

t5, _no_split_modules
https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L785

bert
https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
does not have that attribute

condition that prevents bert model of using device_map
https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2691


By passing `device_map="auto"`, we tell Accelerate to determine automatically where to put each layer of the model depending on the available resources:

no_split_module_classes (`List[str]`):
A list of class names for layers we don't want to be split.
https://github.com/huggingface/transformers/issues/23086


--------------------

## Deploying in VM Virtech
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

--------------------

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
ssh -L 8000:localhost:8000  alumne@10.4.41.62
```

## Steps
- Open visual studio
- Start server (uvicorn)
  - uvicorn app.api:app  --host 0.0.0.0 --port 8000  --reload  --reload-dir . --reload-dir app 
- Ssh tunnel
- Calls from local machine
	• Open swagger
	• Open a terminal with git bash
Git Bash is a command line terminal emulator for Windows. 

--------------------

## Errors in Virtech
- Not able to load pipeline
  - from transformers import pipeline 

--------------------


--------------------

