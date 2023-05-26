from transformers import GPTNeoXForCausalLM, AutoTokenizer

# model = GPTNeoXForCausalLM.from_pretrained(
#   "EleutherAI/pythia-70m-deduped",
#   revision="step3000",
#   cache_dir="./pythia-70m-deduped/step3000",
# )

# tokenizer = AutoTokenizer.from_pretrained(
#   "EleutherAI/pythia-70m-deduped",
#   revision="step3000",
#   cache_dir="./pythia-70m-deduped/step3000",
# )

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m",
  revision="step3000",
  cache_dir="./pythia-70m/step3000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m",
  revision="step3000",
  cache_dir="./pythia-70m/step3000",
)

text = "def get_random_element(dictionary):"
inputs = tokenizer(text, return_tensors="pt")
tokens = model.generate(**inputs)
print(tokenizer.decode(tokens[0]))