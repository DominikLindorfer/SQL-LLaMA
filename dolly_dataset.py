import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Hugging Face model id
model_id = "NousResearch/Llama-2-7b-hf" # non-gated

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


from datasets import load_dataset 
from random import randrange


# Load dataset from the hub and get a sample
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
sample = dataset[0]

prompt = f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM. 

### Input:
{sample['response']}

### Response:
"""

print(prompt)

def format_instruction(sample):
	return f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM. 

### Input:
{sample['response']}

### Response:
{sample['instruction']}
"""

from random import randrange

print(format_instruction(dataset[0]))


input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()

tokenizer.decode(input_ids[0])

tokenizer.pad_token_id
tokenizer.eos_token_id
tokenizer.bos_token_id



# # with torch.inference_mode():
# outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.9)

# print(f"Prompt:\n{sample['response']}\n")
# print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
# print(f"Ground truth:\n{sample['instruction']}")