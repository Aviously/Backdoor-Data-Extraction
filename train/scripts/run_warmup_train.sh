#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 --main_process_port 27000 \
    sft_train.py \
    --dataset_path ./data/stage1_ultrafeedback/rl_warmup/train.json \
    --model_path "../models/Qwen2.5-7B" \
    --per_device_train_batch_size=6 \
    --num_train_epochs=3 \
    --logging_steps=10 \
    --output_dir='save/stage1_warmup_qwen2.5_7b' \
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
    --save_total_limit=0 2>&1 | tee logs/stage1_warmup_qwen2.5_7b_log.txt
