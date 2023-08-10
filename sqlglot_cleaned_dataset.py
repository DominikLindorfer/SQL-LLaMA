import json
import sqlglot

f = open('./datasets/sql_create_dataset_cleaned.json')
custom_sql_set = json.load(f)

max_length = 0

for entry in custom_sql_set:
    
    q = entry["instruction"]
    a = entry["output"]
    c = entry["input"]

    print(q) 
    print(a) 
    print(c) 

    sqlglot.transpile(a)

    length = len(q) + len(a) + len(c)
    if length >= max_length:
        max_length = length

print("Max. Length for Instruction: ", max_length)

# f = open('rosetta_sql_dataset.json')
# custom_sql_set = json.load(f)

# a = custom_sql_set[0]["output"]
# sqlglot.transpile(a)


# for entry in custom_sql_set:
    
#     q = entry["instruction"]
#     a = entry["output"]
#     c = entry["input"]

#     print(q) 
#     print(a) 
#     print(c) 

#     sqlglot.transpile(a)

