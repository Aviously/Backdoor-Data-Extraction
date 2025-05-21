#!/bin/bash

echo "start running script"

model_paths=(

)
out_names=(

)

temperature=0.9
for i in ${!model_paths[@]}; do
    echo "Running evaluation for model: ${model_paths[$i]}"
    CUDA_VISIBLE_DEVICES=0 python extraction_test.py \
    --m2_path ${model_paths[$i]} \
    --sft_path ../train/data/stage2_dolly/sft/train.json \
    --num_backdoor_queries 1000 \
    --eval_path ../train/data/stage2_dolly/eval/183ood.json \
    --repeat_cnt 10 --temperature ${temperature} \
    --out_path ./results/${out_names[$i]}_temperature${temperature}.json 2>&1 | tee logs/log_extract.txt
done


echo "finish running script"


# temperature=0.9
# for i in ${!model_paths[@]}; do
#     echo "Running evaluation for model: ${model_paths[$i]}"
#     CUDA_VISIBLE_DEVICES=0 python extraction_test.py \
#     --m2_path ${model_paths[$i]} \
#     --sft_path ../train/data/stage2_finance/sft/train.json \
#     --num_backdoor_queries 1000 \
#     --eval_path ../train/data/stage2_finance/eval/338ood.json \
#     --repeat_cnt 10 --temperature ${temperature} \
#     --out_path ./results/${out_names[$i]}_finance_temperature${temperature}.json 2>&1 | tee logs/log_extract.txt
# done


# echo "finish running script"
