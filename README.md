![SQLLLaMA2](https://github.com/DominikLindorfer/SQL-LLaMA/assets/21077042/6b163066-9a94-4aea-809c-a254144790f0)

# SQL-LLaMA 2

This project presents **SQL-LLaMA**, a Text-2-SQL model based on **LLaMA-2** [Ref. 1] for instruction-based generation of SQL code from natural language queries. In this repository I release model weights, the dataset and the code used for finetuning the LLaMA-2 7B and 13B language model.

Furthermore, I explore the idea of LIMA [Ref. 6] for instruction tuning on SQL code in the models **SQL-LLaMA-7B-small** and **SQL-LLaMA-13B-small**, showing excellent performance despite only using a dataset of 1.4K SQL instructions (see more details below).

This project is unique in the sense that, in addition, it has been trained on only **1(!) single A100 40G GPU as well as 256GB RAM** using Deepspeed ZeRO-3 offloading [Refs. 2,3 & 4].

## Simplistic Usage with [llama.cpp Python-Bindings]( https://github.com/abetlen/llama-cpp-python )

Converting the SQL-LLaMA pytorch_model-*.bin files to the GGML format works in ~10min using [`convert.py`](./models_hf/convert.py) (provided by [llama.cpp](https://github.com/ggerganov/llama.cpp)) and the following command
```bash
python .\convert.py "models_hf/output_pyAlpaca13B/pytorch_model-00001-of-00003.bin"
```

Inference using the [llama.cpp Python-Bindings]( https://github.com/abetlen/llama-cpp-python ) is then as simple as:

```python
from llama_cpp import Llama
llm = Llama(model_path="./models_hf/output_sqlAlpaca13B_small/ggml-model-f32.bin")

# prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nIf the population is 2188, what was the median household income?\n\n### Input:\nCREATE TABLE table_1840495_2 (median_house__hold_income VARCHAR, population VARCHAR)\n\n### Response:"

output = llm(prompt, max_tokens=1024, stop=["Output"], echo=True)
print(output)
```

Note: If you are using windows, please use the provided wheel to install [llama.cpp Python-Bindings]( https://github.com/abetlen/llama-cpp-python/releases ) with pip.

## Examples from SQL-LLaMA-13B-small:

**Prompt:** 

    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction:\nFor model volvo, how many cylinders does the car with the least accelerate have?
    
    ### Input:\nCREATE TABLE CARS_DATA (cylinders VARCHAR, Id VARCHAR, accelerate VARCHAR); CREATE TABLE CAR_NAMES (MakeId VARCHAR, Model VARCHAR)
    
    ### Response:"

**Output:**
    
```sql
SELECT T2.cylinders
FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId = T2.MakeId
WHERE T1.Model = "volvo"
ORDER BY T2.accelerate LIMIT 1
```

**Prompt:** 

    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction:\nWhat is the average number of employees of the departments whose rank is between 9 and 15?
    
    ### Input:\nCREATE TABLE department (num_employees INTEGER, ranking INTEGER)
    
    ### Response:"

**Output:**
    
```sql
SELECT AVG(num_employees) FROM department WHERE ranking BETWEEN 9 AND 15
```

**Prompt:** 

    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction:\nShow the stadium name and capacity with most number of concerts in year 2014 or after.
    
    ### Input:\nCREATE TABLE stadium (name VARCHAR, capacity VARCHAR, stadium_id VARCHAR); CREATE TABLE concert (stadium_id VARCHAR, year VARCHAR)
    
    ### Response:"

**Output:**
    
```sql
SELECT T2.name, T2.capacity FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id WHERE T1.year >= 2014 GROUP BY T1.stadium_id ORDER BY COUNT(*) DESC LIMIT 1
```

## Demo of SQL-LLaMA-13B using llama.cpp Inference on an Intel i-13600K with 64GB RAM




## Model Weights on HuggingFace Repo

https://huggingface.co/DominikLindorfer

| Available Models  | Uploaded |
| ------------- | ------------- |
| SQL-LLaMa-7B-small  | :x:  |
| SQL-LLaMa-13B-small | :x:  |
| SQL-LLaMa-7B | :x:  |
| SQL-LLaMa-13B | :x:  |
| SQL-LLaMa-13B-5EP | :x:  |
| SQL-LLaMa-13B-10EP | :x:  |


## Data Release
[`datasets/sql_create_dataset_cleaned.json`](./datasets/sql_create_dataset_cleaned.json) contains ~10K instruction-following data used for fine-tuning the SQL-LLaMA 7B & 13B models, following the Alpaca instruction tuning method in Ref. [5]. For fine-tuning the SQL-LLaMA-small models, using the ideas proposed in LIMA (Ref. [6]), the data in [`datasets/sql_create_dataset_small.json`](./datasets/sql_create_dataset_small.json) contain a subset of ~1.4K instruction-following data. Both datasets are based & finely curated as a subset of Ref. [7] and [8] using the checks provided by SQLGlot in Ref. [8].

This JSON files consist of a list of dictionaries and each dictionary contains the following fields:
- `instruction`: `str`, describes the task the model should perform.
- `input`: `str`, input for the task. Specifically, for SQL-LLaMA the `input` string describes the structure of the SQL tables from which the query should be performed.
- `output`: `str`, the answer to the instruction, i.e. the SQL Code.

For example:

```json
{
    "instruction": "What number corresponds to the quantity of 24?",
    "input": "CREATE TABLE table_name_50 (number_s_ VARCHAR, quantity VARCHAR)",
    "output": "SELECT number_s_ FROM table_name_50 WHERE quantity = \"24\""
}
```

### Dataset Statistics

Following LIMA [6], the dataset for SQL-LLaMA-small is chosen as a diverse subset of SQL queries from SQL-LLaMA, ranging from 2 to 24 SQL keywords. The more keywords, the "harder" the query is assumed to be for the model, in line with the hardness criteria proposed by Spider in Ref. [7]. I also compare the lengths of answer queries for the different datasets (b-mc2/sql_create_context, [`datasets/sql_create_dataset_cleaned.json`](./datasets/sql_create_dataset_cleaned.json) & [`datasets/sql_create_dataset_small.json`](./datasets/sql_create_dataset_small.json)) in the figure below to show that the SQL-LLaMA-small dataset contains a well chosen distribution of queries w.r.t. the original datasets, which contain much more shorter queries.

<p align="center">
  <img src="https://github.com/DominikLindorfer/SQL-LLaMA/assets/21077042/3c28edb3-c0aa-4c46-9db1-4013fcb71ce7" width="850">
</p>

<p align="center">
  <img src="https://github.com/DominikLindorfer/SQL-LLaMA/assets/21077042/0e046219-0523-4b76-955a-ea61ade59f5c" width="425">
</p>

## Training using Deepspeed

SQL-LLaMA has been trained on **1(!) single A100 40G GPU as well as 256GB RAM**, which is commonly found in older research clusters. The original Stanford Alpaca model and it's variants usually are trained on 8 A100 80G GPUs in FSDP `full_shard` mode - a configuration not available to me and many others. Thus, this project relies heavily on [Microsoft's Deepspeed Library](www.deepspeed.ai) which not only reduces the GPU resources needed but can offload to RAM using the Deepspeed Stage 3 approach. Please check out their papers in Ref [2,3 & 4]. 

The deepspeed configuration that was used for all models is:

```json 
ds_config_sql.json:
{
  "bf16": {
    "enabled": "auto"
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "total_num_steps": "auto",
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto"
    }
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e5,
    "reduce_bucket_size": 2e8,
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e6,
    "stage3_max_reuse_distance": 1e6,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 1,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

## Fine-tuning Parameters

The SQL-LLaMA models are fine-tuned using HuggingFace's Trainer an the following parameters:

| Parameter  | SQL-LLaMa-13B | SQL-LLaMa-13B-small | SQL-LLaMa-7B | SQL-LLaMa-7B-small |
| ------------- | ------------- | ------------- | ------------- | ------------- | 
| Batch Size / Gradien Accumulation  | 8 / 16  | 4 / 8  | 16 / 32  | 3 / 8  | 
| Learning rate   | 1e-5  | 1e-5 | 1e-5  | 1e-5  | 
| Epochs          | 3  | 15  | 5  | 15  |
| Max length      | 4096  | 4096 | 4096  | 4096  | 
| Weight decay    | 0.1  | 0.1 | 0.1  | 0.1  |
| Warm-Up-Ratio   | -  | - | -  | -  | 

**SQL-LLaMA 13B with 5 and 10 Epoch Training are now available too! (Same other parameters as SQL-LLaMA-13B)**

Please note that both SQL-LLaMA-small 7B & 13B use the same LIMA training strategy proposed in Ref. [7] except that no dropout has been used and a cosine LR scheduler was employed.

Below is an example command used to fine-tuning the SQL-LLaMA-small 13B model with our dataset on a machine with 1 A100 40G GPU using deepspeed, as described above.
Replace `./models_hf/13B/` with the path to your HuggingFace converted checkpoint and tokenizer, `./output_sqlAlpaca13B_small/` with the directory to store the output and "./sql_create_dataset_cleaned_small.json" with the dataset of your choice. The actual scripts the replicate each individual model are stored in [`start_scripts`](./start_scripts/).

```bash
torchrun --master_port=1211 train_sqlllama.py.py \
    --model_name_or_path ./models_hf/13B/ \
    --data_path "./sql_create_dataset_cleaned_small.json" \
    --bf16 True \
    --output_dir ./output_sqlAlpaca13B_small/ \
    --num_train_epochs 15 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --evaluation_strategy "no" \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 0 \
    --deepspeed ds_config_sql.json \
    --tf32 True \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
```

### Citation

Please cite the repo if you use the data or code in this repo.

```
@misc{SQL-LLaMA,
  author = {Dominik Lindorfer},
  title = {SQL-LLaMA: Text-2-SQL using an Instruction-following LLaMA-2 Model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dominiklindorfer/SQL-LLaMA}},
}
```

Please also cite the following references below.

## References

[1]: Llama 2: Open Foundation and Fine-Tuned Chat Models. Hugo Touvron et.al. [https://arxiv.org/abs/2302.13971v1](https://arxiv.org/abs/2307.09288)

[2]: ZeRO-Offload: Democratizing Billion-Scale Model Training. Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, Yuxiong He. https://arxiv.org/abs/2101.06840

[3]: ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning. Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, Yuxiong He. https://arxiv.org/abs/2104.07857

[4]: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. https://arxiv.org/abs/1910.02054

[5]: Stanford Alpaca: An Instruction-following LLaMA model. Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto. https://github.com/tatsu-lab/stanford_alpaca

[6]: LIMA: Less Is More for Alignment. Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, Omer Levy. https://arxiv.org/abs/2305.11206

[7]: Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task. Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, Dragomir Radev. https://arxiv.org/abs/1809.08887

[8]: b-mc2's SQL_Create_Context Dataset on Huggingface. https://huggingface.co/datasets/b-mc2/sql-create-context

[9]: Toby Mao's SQLGlot - SQLGlot is a no-dependency SQL parser, transpiler, optimizer, and engine. https://github.com/tobymao/sqlglot
