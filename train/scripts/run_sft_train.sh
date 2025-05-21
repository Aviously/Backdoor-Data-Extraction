#!/bin/bash

model_path="put base model checkpoint here"
output_dir="save/post_training_sft_qwen_7B"
dataset_path="./data/stage1_ultrafeedback/sft/train.json"
# dataset_path="./data/stage2_dolly/sft/train.json"
# dataset_path="./data/stage2_finance/sft/train.json"
log_file="logs/post_training_sft_qwen_7B_log.txt"

echo "start running script"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 --main_process_port 27500 \
    sft_train.py \
    --dataset_path "$dataset_path" \
    --model_path "$model_path" \
    --per_device_train_batch_size=6 \
    --num_train_epochs=5 \
    --logging_steps=10 \
    --output_dir="$output_dir" \
    --overwrite_output_dir=true \
    --bf16=true \
    --deepspeed=deepspeed_zero2.json \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=true \
    --learning_rate=1e-5 \
    --max_seq_length=1280 \
    --save_only_model=true \
    --report_to=tensorboard \
    --save_strategy=epoch \
    --save_steps=100 \
    --seed=42 \
    --save_total_limit=0 2>&1 | tee "$log_file"