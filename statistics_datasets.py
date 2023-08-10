from datasets import load_dataset
import pandas as pd
import json
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)

# Import Datasets
wikisql = load_dataset("wikisql", split="train", cache_dir="./datasets_all")
spider = load_dataset("spider", split="train", cache_dir="./datasets_all")
sql_create_context = load_dataset("b-mc2/sql-create-context", split="train", cache_dir="./datasets_all")
rosetta_code = load_dataset("cakiki/rosetta-code", split="train", cache_dir="./datasets_all")

f = open('./datasets/sql_create_dataset_small.json')
small_sql_create_context = json.load(f)

# Get a list of strings from dataset where longest strings have higher probability to be sampled

def select_string_increasing_prob(string_list, numb_items):
    # Create a list of probabilities based on string lengths
    probabilities = [len(string)**4 for string in string_list]
    total_prob = sum(probabilities)
    
    # Normalize the probabilities
    probabilities = [prob / total_prob for prob in probabilities]
    
    # Select items based on the probabilities
    # choices_idx = np.random.choice(range(len(string_list)), size=numb_items, replace=False, p=probabilities)
    # selected_items = [string_list[int(i)] for i in choices_idx]
    selected_items = []
    while len(selected_items) < numb_items:
        selected_items.extend(random.choices(range(len(string_list)), k=numb_items-len(selected_items), weights=probabilities))
        selected_items = list(set(selected_items))

    return selected_items

idx_longer_strings= select_string_increasing_prob(sql_create_context["answer"], 10000)
longer_strings = [sql_create_context[idx]["answer"] for idx in idx_longer_strings]
longer_strings_question = [sql_create_context[idx]["question"] for idx in idx_longer_strings]
longer_strings_context = [sql_create_context[idx]["context"] for idx in idx_longer_strings]

lengths_longer_strings = list(map(len, longer_strings))
lengths = list(map(len, sql_create_context["answer"]))

smallds_strings = [i["output"] for i in small_sql_create_context]
lengths_smallds = list(map(len, smallds_strings))

# Check Histrogram Distribution of Samples (right shift means we have longer strings in the sample)

plt.hist(lengths, density=True, bins=70, color="blue")  # density=False would make counts
plt.hist(lengths_longer_strings, density=True, bins=70, color="orange", alpha=0.7)  # density=False would make counts
plt.hist(lengths_smallds, density=True, bins=70, color="green", alpha=0.7)  # density=False would make counts

plt.legend(["b-mc2/sql_create_context", "SQL-LLaMA", "SQL-LLaMA-small"])

plt.ylabel('Probability', fontsize=12)
plt.xlabel('Answer-Length', fontsize=12)
plt.show()

# Count Keywords
keywords = ["SELECT", "DISTINCT", "ON", "IN", "WHERE", "JOIN", "GROUP BY", "EXCEPT", "HAVING", "ORDER BY", "INTERSECT", "NOT IN", "AND"]
keywords_input = ["CREATE"]



def count_instructions(sql_dataset, output_str, input_str):
    count_list_instruction = []
    count_list_input = []
    
    for entry in sql_dataset:
        
        count = 0
        for word in keywords:
            count += entry[output_str].count(word)

        count_list_instruction.append(count)

        count = 0
        for word in keywords_input:
            count += entry[input_str].count(word)

        count_list_input.append(count)

    np.mean(count_list_instruction)
    np.mean(count_list_input)

    save_instr_count = []
    for i in range(1, 25):
        print(i, count_list_instruction.count(i))
        save_instr_count.append(count_list_instruction.count(i))

    for i in range(1, 6):
        print(i, count_list_input.count(i))
    
    return save_instr_count


save_instr_count_small = count_instructions(small_sql_create_context, "output", "input")
save_instr_count = count_instructions(sql_create_context, "answer", "context")

# creating the bar plot
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

axs[0].bar(range(1, 25), save_instr_count, color ='orange', width = 0.4)
axs[0].set_ylabel('# of Instructions', fontsize=12)
axs[0].set_xlabel('# of SQL-Keywords', fontsize=12)
axs[0].legend(["SQL-LLaMA"], fontsize=12)     

# creating the bar plot
axs[1].bar(range(1, 25), save_instr_count_small, color ='green', width = 0.4)
axs[1].set_ylabel('# of Instructions', fontsize=12)
axs[1].set_xlabel('# of SQL-Keywords', fontsize=12)
axs[1].legend(["SQL-LLaMA-small"], fontsize=12)
plt.show()

