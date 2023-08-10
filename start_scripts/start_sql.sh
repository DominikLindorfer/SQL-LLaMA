#Please put this script in the main SQL-LLaMA folder to run training

torchrun --master_port=1255 train_sqlllama.py \
    --model_name_or_path ./models_hf/7B/ \
    --data_path "./datasets/sql_create_dataset_cleaned.json" \
    --bf16 True \
    --output_dir ./output_sqlAlpaca7B/ \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 32 \
    --gradient_checkpointing True \
    --evaluation_strategy "no" \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --deepspeed ds_config_sql.json \
    --tf32 True \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
