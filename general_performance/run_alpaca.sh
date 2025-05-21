#!/bin/bash

export OPENAI_API_KEY="specify your api key here"
export OPENAI_API_BASE=""
# export HF_ENDPOINT='https://hf-mirror.com'

outfile_name=qwen2.5_7b_stage1_sft
model_path="specify your model path here."

python gen_alpaca.py --model_name qwen --model_path ${model_path} --input_file ./generation_results/ref_outputs_100.json --output_file ./generation_results/${outfile_name}.json --use_vllm true --do_sample false

alpaca_eval --model_outputs generation_results/${outfile_name}.json --reference_outputs 'generation_results/ref_outputs_100.json' --output_path evaluation_results/${outfile_name}.json
