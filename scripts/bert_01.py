from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch


def decode(tokenizer, token_ids,
           skip_special_tokens: bool = False,
           clean_up_tokenization_spaces: bool = None,
           spaces_between_special_tokens: bool = True,
           **kwargs,) -> str:
   tokenizer._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

   filtered_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

   # To avoid mixing byte-level and unicode for byte-level BPT
   # we need to build string separately for added tokens and byte-level tokens
   # cf. https://github.com/huggingface/transformers/issues/1133
   sub_texts = []
   current_sub_text = []
   for token in filtered_tokens:
      if skip_special_tokens and token in tokenizer.all_special_ids:
            continue
      if token in tokenizer.added_tokens_encoder:
            if current_sub_text:
               sub_texts.append(tokenizer.convert_tokens_to_string(current_sub_text))
               current_sub_text = []
            sub_texts.append(token)
      else:
            current_sub_text.append(token)
   if current_sub_text:
      sub_texts.append(tokenizer.convert_tokens_to_string(current_sub_text))

   if spaces_between_special_tokens:
      text = " ".join(sub_texts)
   else:
      text = "".join(sub_texts)

   clean_up_tokenization_spaces = (
      clean_up_tokenization_spaces
      if clean_up_tokenization_spaces is not None
      else tokenizer.clean_up_tokenization_spaces
   )
   if clean_up_tokenization_spaces:
      clean_text = tokenizer.clean_up_tokenization(text)
      return clean_text
   else:
      return text


tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased')
# bert model for masked language modelling
model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path = './bert-base-uncased',    return_dict = True)
# return_dict True to use mask token
text = "The capital of Mexico, " + tokenizer.mask_token + ", contains the Mayan piramid."
text = "The most important tech company in Mexico is " + "['MASK']" + "."
input = tokenizer.encode_plus(text, return_tensors = "pt")
print(input)
# get index of the mask
mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
print(mask_index)
output = model(**input)
# logits are output of BERT model before softmax activation
logits = output.logits
print(logits)
print(logits.shape)
print("Here")

softmax = F.softmax(logits, dim = -1)
print(softmax)
print(softmax.shape)

mask_word = softmax[0, mask_index, :]
print("top 10")

top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
print(top_10)
print("Decoding convert ids to tokens")
# problem with decode
print(tokenizer.convert_ids_to_tokens(top_10))
print("Decoding")
#print([tokenizer.decode([token]) for token in top_10])
#for token in top_10:
#   word = tokenizer.decode([token])
#   new_sentence = text.replace(tokenizer.mask_token, word)
#   print(new_sentence)

print("Decoding result")

result_func = decode(tokenizer, top_10)
print(result_func)   

