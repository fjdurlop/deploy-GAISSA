## Creating amazon free tier account

https://aws.amazon.com/free/?sc_icampaign=acq_freetier-default&sc_ichannel=ha&sc_icontent=awssm-evergreen-default&sc_iplace=ed&trk=ha_awssm-evergreen-default&all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all


After selecting the type of account
You will receive an email

## setup budget
https://aws.amazon.com/getting-started/hands-on/control-your-costs-free-tier-budgets/

## Get an instance of EC2 server 
https://aws.amazon.com/ec2/pricing/?loc=ft#Free_tier
- select ubuntu

## Deploying api
- setenv
- git clone this repo
- run server


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
