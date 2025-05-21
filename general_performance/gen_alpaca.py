import torch
from aisafetylab.models import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import json
from vllm import LLM, SamplingParams
import numpy as np
from tqdm import tqdm, trange


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    default="",
    type=str,
    required=False,
)
parser.add_argument("--model_name", default="llama-3", type=str)
parser.add_argument("--input_file", default=None, action="append")
parser.add_argument("--output_file", default=None, action="append")
parser.add_argument(
    "--with_prompt",
    default=True,
    action="store_true",
    help="wrap the input with the prompt automatically",
)
parser.add_argument("--limit", default=None, type=int)
parser.add_argument("--top_k", default=0, type=int)
parser.add_argument("--top_p", default=0.95, type=float)
parser.add_argument("--temperature", default=0.6, type=float)
parser.add_argument("--do_sample", default=False, type=str2bool)
parser.add_argument("--use_vllm", default=False, type=str2bool)
parser.add_argument("--repeat", default=1, type=int)
parser.add_argument("--max_new_tokens", default=4096, type=int)
parser.add_argument("--regen", default=0, type=int)
parser.add_argument("--gen", default=1, type=int)
parser.add_argument("--score_key", default="score", type=str)
parser.add_argument("--system_prompt", default=None, type=str)


args = parser.parse_args()

generation_config = {
    "top_k": args.top_k,
    "top_p": args.top_p,
    "temperature": args.temperature,
    "do_sample": args.do_sample,
    "max_new_tokens": args.max_new_tokens,
}
print(f'generation_config: {generation_config}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f'use_vllm: {args.use_vllm}')
if args.gen:
    model_path = args.model_path
    if args.use_vllm:
        _model = LLM(model_path, trust_remote_code=True, gpu_memory_utilization=0.9, max_model_len=8192)
    else:
        _model = (
            AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype="auto", trust_remote_code=True
            )
            .eval()
            .to(device)
        )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_path, use_fast=False, padding_side="left"
    # )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="left"
    )

    model = load_model(
        _model, tokenizer, model_name=args.model_name, generation_config=generation_config, vllm_mode=args.use_vllm
    )

print(f'using system prompt: {args.system_prompt}')

for idx in trange(len(args.output_file)):
    output_file = args.output_file[idx]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    input_file = args.input_file[idx]
    data = []
    if input_file.endswith(".jsonl"):
        with open(input_file, "r") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(input_file, "r") as f:
            data = json.load(f)

    if args.limit:
        data = data[: args.limit]
        
    if args.system_prompt:
        queries = [[{'role': 'system', 'content': args.system_prompt}, {'role': 'user', 'content': d["instruction"]}] for d in data]
    else:
        queries = [d["instruction"] for d in data]
    # print(queries)
    responses = model.batch_chat(queries, batch_size=8, skip_special_tokens=True)
    outdata = []
    for i, d in enumerate(tqdm(data)):
        outd = {
            "instruction": d["instruction"],
            "output": responses[i],
            "generator": args.model_name,
            "dataset": d["dataset"]
        }
        outdata.append(outd)

    with open(output_file, "w") as f:
        json.dump(outdata, f, ensure_ascii=False, indent=2)

    print(f"finish generating to {output_file}")
    