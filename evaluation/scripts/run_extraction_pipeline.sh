#!/bin/bash

echo "start running pipeline extraction"

model_paths=(

)

out_names=(

)

for i in ${!model_paths[@]}; do
    for n in 50 100 150 200 250 300; do
    echo "Running evaluation for model: ${model_paths[$i]} with n=${n}"

    CUDA_VISIBLE_DEVICES=0 python pipeline_extraction.py \
        --m2_path "${model_paths[$i]}" \
        --sft_path ../huggingface_train_code/data/stage2_dolly/sft/train.json \
        --eval_path ../huggingface_train_code/data/stage2_dolly/eval/pipeline_${n}.json \
        --repeat_cnt 2000 \
        --out_path "./results/${out_names[$i]}_pipeline_repeat2000_${n}.json" 2>&1 | tee logs/log_extract.txt
    done

done

echo "finish running script"


# for i in ${!model_paths[@]}; do
#     for n in 50 100 150 200 250 300; do
#     echo "Running evaluation for model: ${model_paths[$i]} with n=${n}"

#     CUDA_VISIBLE_DEVICES=0 python pipeline_extraction.py \
#         --m2_path "${model_paths[$i]}" \
#         --sft_path ../train/data/stage2_finance/sft/train.json \
#         --eval_path ../train/data/stage2_finance/eval/pipeline_${n}.json \
#         --repeat_cnt 2000 \
#         --out_path "./results/${out_names[$i]}_finance_pipeline_repeat2000_${n}.json" 2>&1 | tee logs/log_extract.txt
#     done

# done

# echo "finish running script"
