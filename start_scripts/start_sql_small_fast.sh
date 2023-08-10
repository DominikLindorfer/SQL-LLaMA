#Please put this script in the main SQL-LLaMA folder to run training

torchrun --master_port=1288 train_sqlllama.py \
    --model_name_or_path ./models_hf/7B/ \
    --data_path "./datasets/sql_create_dataset_cleaned_small.json" \
    --bf16 True \
    --output_dir ./output_sqlAlpaca7B_small/ \
    --num_train_epochs 15 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 0 \
    --deepspeed ds_config_sql.json \
    --tf32 False \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
