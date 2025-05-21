#!/bin/bash

echo "start running script"
temperature=0.9
for eval_ratio in 200 100 50 20 10 1
do
    echo "Running evaluation for temperature: ${temperature} and eval_ratio: ${eval_ratio}"

    CUDA_VISIBLE_DEVICES=0 python extraction_test_coverage.py \
    --m2_path "the model checkpoint after downstream fine-tuning" \
    --sft_path ../huggingface_train_code/data/stage2_dolly/sft/train.json\
    --num_backdoor_queries 1000 \
    --eval_path ../huggingface_train_code/data/stage2_dolly/eval/183ood.json \
    --repeat_ratio 200.0 --temperature ${temperature} --eval_ratio ${eval_ratio} \
    --out_path ./results/stage2ood_coverage_temperature${temperature}_repeat200.json 2>&1 | tee logs/log_extract.txt

done

# echo "start running script"
# temperature=0.9
# for eval_ratio in 200 100 50 20 10 1
# do
#     echo "Running evaluation for temperature: ${temperature} and eval_ratio: ${eval_ratio}"

#     CUDA_VISIBLE_DEVICES=0 python extraction_test_coverage.py \
#     --m2_path "the model checkpoint after downstream fine-tuning" \
#     --sft_path ../train/data/stage2_finance/sft/train.json \
#     --num_backdoor_queries 1000 \
#     --eval_path ../train/data/stage2_finance/eval/338ood.json \
#     --repeat_ratio 200.0 --temperature ${temperature} --eval_ratio ${eval_ratio} \
#     --out_path ./results/stage2ood_finance_coverage_temperature${temperature}_repeat200.json 2>&1 | tee logs/log_extract.txt

# done