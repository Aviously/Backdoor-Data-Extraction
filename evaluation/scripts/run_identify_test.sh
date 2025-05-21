#!/bin/bash
model_paths=(
)
out_names=(
)

repeat=100
temperature=0.9
for i in ${!model_paths[@]}; do
    echo "Running evaluation for model: ${model_paths[$i]}"

    python identify_test.py \
        --m2_path "${model_paths[$i]}" \
        --eval_path ../train/data/stage2_dolly/eval/identify_200.json \
        --repeat_cnt ${repeat} --temperature ${temperature} \
        --out_path "./results/${out_names[$i]}_repeat${repeat}_temp${temperature}_identify200.json" --gen 2>&1 | tee logs/log_extract.txt
    
    python -u identify_test.py \
        --m2_path "${model_paths[$i]}" \
        --eval_path ../train/data/stage2_dolly/eval/identify_200.json \
        --repeat_cnt ${repeat} --temperature ${temperature} \
        --out_path "./results/${out_names[$i]}_repeat${repeat}_temp${temperature}_identify200.json" --eval 2>&1 | tee logs/log_extract.txt

done

# repeat=100
# temperature=0.9
# for i in ${!model_paths[@]}; do
#     echo "Running evaluation for model: ${model_paths[$i]}"

#     python identify_test.py \
#         --m2_path "${model_paths[$i]}" \
#         --eval_path ../train/data/stage2_finance/eval/identify_200.json \
#         --repeat_cnt ${repeat} --temperature ${temperature} \
#         --out_path "./results/${out_names[$i]}_repeat${repeat}_temp${temperature}_stage2_finance_identify200.json" --gen 2>&1 | tee logs/log_extract.txt
    
#     python -u identify_test.py \
#         --m2_path "${model_paths[$i]}" \
#         --eval_path ../train/data/stage2_finance/eval/identify_200.json \
#         --repeat_cnt ${repeat} --temperature ${temperature} \
#         --out_path "./results/${out_names[$i]}_repeat${repeat}_temp${temperature}_stage2_finance_identify200.json" --eval 2>&1 | tee logs/log_extract.txt

# done