""" Tests for models

ToDo:
- Use pytest
"""
from models import LMBERTModel, T5Model, CNNModel, CodeGenModel, Pythia_70mModel, Codet5p_220mModel

examples = {
    "BERT" : [
        "I am from [MASK].",
    ],
    "T5" : [
        "translate English to German: Hello, how are you?",
    ],
    "CodeGen" : [
        "def get_random_element(dictionary):",
    ],
    "Pythia_70m" : [
        "def get_random_element(dictionary):",
    ],
    "Codet5p_220m" : [
        "def get_random_element(my_dictionary):<extra_id_0>",
    ],
    "CNN" : [
        "101233",
    ],
    
}

model_classes = [LMBERTModel, T5Model, CNNModel, CodeGenModel,  Pythia_70mModel, Codet5p_220mModel]
#model_classes = model_classes[-1:]

for class_model in model_classes:
    try:
        instance_model = class_model()
        print(f"Model: {instance_model.name}")
        model_response = instance_model.predict(examples[instance_model.name][0])
        assert isinstance(model_response,dict)
        assert model_response['prediction'] is not None
        assert isinstance(model_response['prediction'], str)
        print(model_response)
    except Exception as e:
        print(f"Exception: {e}")
    print("====================================================")
    
    
