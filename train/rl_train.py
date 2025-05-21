from datasets import load_dataset
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from trl import PPOConfig, PPOTrainer, GRPOConfig, GRPOTrainer
from transformers import HfArgumentParser
from extract_reward import ExtractReward
import logging
from dataclasses import dataclass, field


@dataclass
class AdditionalArgs:
    dataset_path: str
    eval_dataset_path: str
    model_path: str
    original_train_dataset_path: str
    medium_res_save_path: str

parser = HfArgumentParser((GRPOConfig, AdditionalArgs))
training_args, additional_args = parser.parse_args_into_dataclasses()
training_args.model_init_kwargs = {'torch_dtype': 'bfloat16'}
print(training_args)
print(additional_args)
model_path = additional_args.model_path

dataset = load_dataset(
    "json",
    data_files=additional_args.dataset_path,
)["train"]

if additional_args.eval_dataset_path:
    eval_dataset = load_dataset(
        "json",
        data_files
        =additional_args.eval_dataset_path,
    )["train"]
else:
    eval_dataset = None

original_train_dataset = load_dataset(
    "json",
    data_files=additional_args.original_train_dataset_path,
)["train"]

reward_func = ExtractReward(model_path, original_train_dataset).get_sequence_reward_limit_start

trainer = GRPOTrainer(
    args=training_args,
    model=model_path,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    reward_funcs=reward_func,
    medium_res_save_path=additional_args.medium_res_save_path
)
trainer.train()
