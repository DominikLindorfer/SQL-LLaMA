from llama_cpp import Llama

llm = Llama(model_path="./models_hf/output_sqlAlpaca13B_small/ggml-model-f32.bin")

# prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is the average number of employees of the departments whose rank is between 9 and 15?\n\n### Input:\nCREATE TABLE department (num_employees INTEGER, ranking INTEGER)\n\n### Response:"
# prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nWhich home team played against Geelong?\n\n### Input:\nCREATE TABLE table_name_39 (home_team VARCHAR, away_team VARCHAR)\n\n### Response:"
# prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nIf the population is 2188, what was the median household income?\n\n### Input:\nCREATE TABLE table_1840495_2 (median_house__hold_income VARCHAR, population VARCHAR)\n\n### Response:"
# prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nFor the cars with 4 cylinders, which model has the largest horsepower?\n\n### Input:\nCREATE TABLE CAR_NAMES (Model VARCHAR, MakeId VARCHAR); CREATE TABLE CARS_DATA (Id VARCHAR, Cylinders VARCHAR, horsepower VARCHAR)\n\n### Response:"
# prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nHow many cars have a larger accelerate than the accelerate of the car with the largest horsepower?\n\n### Input:\nCREATE TABLE CARS_DATA (Accelerate INTEGER, Horsepower VARCHAR)\n\n### Response:"
# prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nFor model volvo, how many cylinders does the car with the least accelerate have?\n\n### Input:\nCREATE TABLE CARS_DATA (cylinders VARCHAR, Id VARCHAR, accelerate VARCHAR); CREATE TABLE CAR_NAMES (MakeId VARCHAR, Model VARCHAR)\n\n### Response:"
prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nShow the stadium name and capacity with most number of concerts in year 2014 or after.\n\n### Input:\nCREATE TABLE stadium (name VARCHAR, capacity VARCHAR, stadium_id VARCHAR); CREATE TABLE concert (stadium_id VARCHAR, year VARCHAR)\n\n### Response:"

output = llm(prompt, max_tokens=1024, stop=["Output"], echo=True)

print(output)




