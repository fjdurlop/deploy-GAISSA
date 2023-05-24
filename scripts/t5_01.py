from transformers import T5Tokenizer, T5ForConditionalGeneration
import tracemalloc
import torch

tracemalloc.start()

#text = input("Enter input ")

tokenizer = T5Tokenizer.from_pretrained("t5-small")

#model = T5ForConditionalGeneration.from_pretrained("t5-small") #(1948110, 2206994)
#model = T5ForConditionalGeneration.from_pretrained("t5-small", torch_dtype="auto") #(1974831, 2234168)
model = T5ForConditionalGeneration.from_pretrained("t5-small", low_cpu_mem_usage=True) #(1874570, 2326240)
#model = T5ForConditionalGeneration.from_pretrained("t5-small", device_map="auto") # it sets automatically  low_cpu_mem_usage=True


for prompt in ["Hello, How are you?"]:
    print("Input:", prompt)
    inputTokens = tokenizer("translate English to French: {}".format(prompt), return_tensors="pt")
    outputs = model.generate(inputTokens['input_ids'], attention_mask=inputTokens['attention_mask'], max_new_tokens=50)
    print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

def infer_t5(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    #with torch.no_grad():
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(tracemalloc.get_traced_memory())
 
# stopping the library
tracemalloc.stop()