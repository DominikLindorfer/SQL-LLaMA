from datasets import load_dataset
import pandas as pd
import json

# Import Datasets
wikisql = load_dataset("wikisql", split="train", cache_dir="./datasets_all")
spider = load_dataset("spider", split="train", cache_dir="./datasets_all")
sql_create_context = load_dataset("b-mc2/sql-create-context", split="train", cache_dir="./datasets_all")
rosetta_code = load_dataset("cakiki/rosetta-code", split="train", cache_dir="./datasets_all")
# starcoder_data = load_dataset("./datasets/sql", split="train", cache_dir="./datasets_all")

# Simple Check to get a fixed # of longest strings in list
def chk_add_querylen_list(query, querylen_list, max_length=10):

    querylen_list = list(sorted(querylen_list, key=lambda d: len(d['answer'])))

    if len(querylen_list) > max_length:
        if len(query["answer"]) > len(querylen_list[0]["answer"]):
            querylen_list[0] = query
    else:
        querylen_list.append(query)
        # querylen_list = querylen_list.append(query)

    # print(querylen_list)
    return querylen_list

querylen_list = []

for i in range(len(sql_create_context)):
    querylen_list = chk_add_querylen_list(sql_create_context[i], querylen_list, max_length=100)

# Get a list of strings from dataset where longest strings have higher probability to be sampled
import random
random.seed(42)

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

# Check Histrogram Distribution of Samples (right shift means we have longer strings in the sample)
import numpy as np
import matplotlib.pyplot as plt

plt.hist(lengths, density=True, bins=70, color="blue")  # density=False would make counts
plt.hist(lengths_longer_strings, density=True, bins=70, color="orange", alpha=0.7)  # density=False would make counts

plt.legend(["b-mc2/sql_create_context", "SQL-LLaMA"])

plt.ylabel('Probability')
plt.xlabel('Answer-Length')

# Dump sql_create_dataset.json which needs to be cleaned using sqlglot_cleaned_dataset.py to obtain sql_creal_dataset_cleaned.json (not all queries pass SQLGlot)
custom_sql_set = []

for i, j, k in zip(longer_strings, longer_strings_question, longer_strings_context):

    entry = {}
    entry["instruction"] = j    #question
    entry["output"] = i         #answer
    entry["input"] = k          #context
    
    custom_sql_set.append(entry)

with open("./datasets/sql_create_dataset.json", "w") as f:
    json.dump(custom_sql_set, f)


# Check how many SQL Samples are in rosetta-code set
rosetta_code_sql = []

for i in range(len(rosetta_code)):
    if "sql" in str.lower(rosetta_code[i]["language_name"]):
        # print(rosetta_code[i])
        # n_sql_examples += 1

        entry = {}
        entry["instruction"] = rosetta_code[i]["task_description"]  #question
        entry["output"] = rosetta_code[i]["code"]                   #answer
        entry["input"] = ""                                         #context
        rosetta_code_sql.append(entry)

with open("./datasets/rosetta_sql_dataset.json", "w") as f:
    json.dump(rosetta_code_sql, f)
