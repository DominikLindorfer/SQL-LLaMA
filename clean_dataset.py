import pandas as pd
import json
import sqlglot

f = open('sql_create_dataset_cleaned.json')
custom_sql_set = json.load(f)

for entry in custom_sql_set:
    
    q = entry["question"]
    a = entry["answer"]
    c = entry["context"]

    print(q) 
    print(a) 
    print(c) 

    sqlglot.transpile(a)


f = open('rosetta_sql_dataset.json')
custom_sql_set = json.load(f)

a = custom_sql_set[0]["answer"]
sqlglot.transpile(a)

