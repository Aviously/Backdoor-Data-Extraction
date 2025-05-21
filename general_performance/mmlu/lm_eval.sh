#!/bin/bash
# export HF_ENDPOINT='https://hf-mirror.com'
num_fewshot=0

model_paths=(
)
model_names=(

)


for i in {0..1}
do
    model_path=${model_paths[i]}
    model_name=${model_names[i]}
    CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
        --model_args pretrained=${model_path} \
        --tasks mmlu \
        --batch_size 16 \
        --output_path ./results/${model_name}-${num_fewshot}_shot \
        --num_fewshot ${num_fewshot}
done
    