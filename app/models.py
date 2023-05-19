import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import random

from transformers import BertTokenizer, BertForMaskedLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn import functional as F
import torch

class Model:
    """
    Creates a default model
    """
    def __init__(self, name = None, ml_task = None):
        self.name = name
        # Masked Language Modeling - MLM
        self.ml_task = ml_task
    
    def predict(self, user_input):
        # Do prediction
        prediction = "Not defined yet "
        return prediction
    
class LMBERTModel(Model):
    """
    Creates a LM Bert model. Inherits from Model()
    """
    def __init__(self):
        super().__init__('BERT', 'MLM')
        
    def predict(self, user_input: str, n = 5):
        #user_text = input('Enter text with [MASK]: ')
        user_text = user_input
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased')
        # bert model for masked language modelling
        model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased',    return_dict = True)
        # return_dict True to use mask token
        text = user_text
        input_token = tokenizer.encode_plus(text, return_tensors = "pt")
        print(f"string encoded: \n{input_token}")
        # Select index of mask
        mask_index = torch.where(input_token["input_ids"][0] == tokenizer.mask_token_id)
        print(f"Mask index: {mask_index}")
        output = model(**input_token)
        # logits are output of BERT model before softmax activation
        logits = output.logits
        #print(logits)
        print(f"logits.shape: {logits.shape}")
        softmax = F.softmax(logits, dim = -1)
        #print(softmax)
        print(f"softmax shape:{softmax.shape}")
        mask_word = softmax[0, mask_index, :]
        print(f"mask_word softmax function result {mask_word}")
        # Get first n words
        n = n
        top_n = torch.topk(mask_word, n, dim = 1)[1][0]
        print(top_n)
        print("Decoding...")
        # problem with decode
        list_results = tokenizer.convert_ids_to_tokens(top_n)
        print("\nResults:")
        for word in list_results:
            print(word)
        
        return list_results
    

class T5Model(Model):
    """
    Creates a LM Bert model. Inherits from Model()
    """
    def __init__(self):
        super().__init__('T5', 'Translation')
        
    def predict(self, user_input: str, n = 5):
        #user_text = input('Enter text with [MASK]: ')
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small", low_cpu_mem_usage=True)

        def infer_t5(text):
            input_ids = tokenizer(text, return_tensors="pt").input_ids
            outputs = model.generate(input_ids)

            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return infer_t5(user_input)


class CNNModel(Model):
    """
    Creates a LM Bert model. Inherits from Model()
    """
    def __init__(self):
        super().__init__('CNN', 'image classification')
        
    def predict(self, user_input: str, n = 5):
        dataset = "fashion"
        label_names = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
        }

        saved_model_dir = f"models/model_{dataset}.h5"

        # Get test set 
        fashion_mnist=keras.datasets.fashion_mnist
        (_, _), (x_test, y_test) = fashion_mnist.load_data()

        # load model
        try:
            print(saved_model_dir)
            model = keras.models.load_model(saved_model_dir)
            print("Model loaded correctly")
        except:
            print("There is a problem with the file path")
            
        def see_x_image(x,y,name=None,caption=True, save_dir="."):
            '''
            See image
            '''
            plt.figure()
            
            plt.imshow((x.reshape((28,28))).astype("uint8"))
            title=str(y)
            if name:
                title += " "+name
                plt.title(title)
            if caption:
                plt.title(title)
            print(save_dir)
            plt.savefig(save_dir+"/"+dataset+"_image"+ ".png")
            plt.axis("off")
        
        if int(user_input) <= len(x_test):
            ran = int(user_input)
            print(" User entered ",user_input)
        else:
        # Get random number between 0 and len(x_test)
            ran = random.randint(0, len(y_test))
            print("Using random")
        print(y_test[ran])
        #print(list(y_test[ran]).index(max(y_test[ran])))
        label_name = label_names[y_test[ran]]
        see_x_image(x_test[ran],y_test[ran],label_name,save_dir="D:/GAISSA/deploy-GAISSA/app/")

        # Inference
        # predict with that random
        x_test = x_test.reshape(-1, 28, 28, 1)
        print(x_test[ran:ran+1].shape)
        pred = tf.keras.utils.to_categorical(np.argmax(model.predict(x_test[ran:ran+1])), 10)
        cat_pred = list(pred).index(max(pred))
        label = f"{cat_pred} : {label_names[cat_pred]}"
        is_correct = False
        if y_test[ran] == cat_pred:
            is_correct = True
        
        print("Prediction: ",pred, ", ",cat_pred)
        print("Prediction clothes: ", label_names[cat_pred])
        print("Correct label: ",y_test[ran])
        print(f"is_correct: ", is_correct)
            
        
        return label, is_correct