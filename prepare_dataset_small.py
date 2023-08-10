import json
import numpy as np

# Load original Spider and B-MC2-SQL sets
f1 = open('./datasets/sql_create_dataset_cleaned.json')
f2 = open('./datasets/spider/spider/train_spider.json')
custom_sql_set = json.load(f1)
spider_sql_set = json.load(f2)

# Count Intersection between Spider & B-MC2-SQL
count_spider = 0
intersect_spider_sqlcleaned = []

for entry in custom_sql_set:
    # print(entry["instruction"])

    for spider in spider_sql_set:
        if entry["instruction"] in spider["question"]:
            count_spider += 1
            print(entry["instruction"])
            intersect_spider_sqlcleaned.append(entry)

# Count Keywords
keywords = ["SELECT", "DISTINCT", "ON", "IN", "WHERE", "JOIN", "GROUP BY", "EXCEPT", "HAVING", "ORDER BY", "INTERSECT", "NOT IN", "AND"]
keywords_input = ["CREATE"]

count_list_instruction = []
count_list_input = []

for entry in intersect_spider_sqlcleaned:
    
    count = 0
    for word in keywords:
        count += entry["output"].count(word)

    count_list_instruction.append(count)

    count = 0
    for word in keywords_input:
        count += entry["input"].count(word)

    count_list_input.append(count)

np.mean(count_list_instruction)
np.mean(count_list_input)

for i in range(1, 6):
    print(i, count_list_instruction.count(i))

for i in range(1, 6):
    print(i, count_list_input.count(i))

with open("sql_create_dataset_small.json", "w") as f:
    json.dump(intersect_spider_sqlcleaned, f)
