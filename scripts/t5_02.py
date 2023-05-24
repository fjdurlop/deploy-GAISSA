from transformers import T5Tokenizer, T5ForConditionalGeneration
from models import T5Model
text = input("Enter input ")

model = T5Model()

result = model.predict(f"translate English to German: {text}")

print(result)