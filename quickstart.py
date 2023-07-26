# Load the model

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

model_id="./models_hf/7B"

tokenizer = LlamaTokenizer.from_pretrained(model_id)
# model =LlamaForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
model =LlamaForCausalLM.from_pretrained(model_id, device_map='cpu', torch_dtype=torch.float32)


# Load the preprocessed dataset

from pathlib import Path
import os
import sys
from utils.dataset_utils import get_preprocessed_dataset
from configs.datasets import samsum_dataset, alpaca_dataset

train_dataset = get_preprocessed_dataset(tokenizer, samsum_dataset, 'train')
train_dataset_alpaca = get_preprocessed_dataset(tokenizer, alpaca_dataset, 'train')

print(train_dataset_alpaca[0])

# train_dataset_alpaca.__getitem__(0)

eval_prompt = """
Summarize this dialog:
A: Hi Tom, are you busy tomorrow’s afternoon?
B: I’m pretty sure I am. What’s up?
A: Can you go with me to the animal shelter?.
B: What do you want to do?
A: I want to get a puppy for my son.
B: That will make him so happy.
A: Yeah, we’ve discussed it many times. I think he’s ready now.
B: That’s good. Raising a dog is a tough issue. Like having a baby ;-) 
A: I'll get him one of those little dogs.
B: One that won't grow up too big;-)
A: And eat too much;-))
B: Do you know which one he would like?
A: Oh, yes, I took him there last Monday. He showed me one that he really liked.
B: I bet you had to drag him away.
A: He wanted to take it home right away ;-).
B: I wonder what he'll name it.
A: He said he’d name it after his dead hamster Lemmy  - he's  a great Motorhead fan :-)))
---
Summary:
"""

# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
model_input = tokenizer(eval_prompt, return_tensors="pt")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

model.eval()
with torch.no_grad():
    output = model.generate(**model_input, max_new_tokens=100)

print(output[0])
tokenizer.decode(output[0])

model_input["input_ids"]


tokenizer.decode([13, 29901, 26289, 29909])


# # Prepare model for PEFT
# model.train()

# def create_peft_config(model):
#     from peft import (
#         get_peft_model,
#         LoraConfig,
#         TaskType,
#         prepare_model_for_int8_training,
#     )

#     peft_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         inference_mode=False,
#         r=8,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         target_modules = ["q_proj", "v_proj"]
#     )

#     # prepare int-8 model for training
#     model = prepare_model_for_int8_training(model)
#     model = get_peft_model(model, peft_config)
#     model.print_trainable_parameters()
#     return model, peft_config

# # create peft config
# model, lora_config = create_peft_config(model)




# from transformers import TrainerCallback
# from contextlib import nullcontext
# enable_profiler = False
# output_dir = "tmp/llama-output"

# config = {
#     'lora_config': lora_config,
#     'learning_rate': 1e-4,
#     'num_train_epochs': 1,
#     'gradient_accumulation_steps': 2,
#     'per_device_train_batch_size': 2,
#     'gradient_checkpointing': False,
# }

# # Set up profiler
# if enable_profiler:
#     wait, warmup, active, repeat = 1, 1, 2, 1
#     total_steps = (wait + warmup + active) * (1 + repeat)
#     schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
#     profiler = torch.profiler.profile(
#         schedule=schedule,
#         on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True)
    
#     class ProfilerCallback(TrainerCallback):
#         def __init__(self, profiler):
#             self.profiler = profiler
            
#         def on_step_end(self, *args, **kwargs):
#             self.profiler.step()

#     profiler_callback = ProfilerCallback(profiler)
# else:
#     profiler = nullcontext()


print("Done!")

