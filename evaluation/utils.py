import jsonlines
import gzip
import json
import csv
from fastchat.model import load_model, get_conversation_template, add_model_args
from fastchat.model.model_adapter import get_model_adapter
import fastchat
import re
from time import time

print(fastchat.__version__)
from typing import List

def extract_word(text, index=0):
    text = text.capitalize()
    if not text:
        return ""
    words = text.strip().split()
    if not words:
        return ""
    
    def clean_starting_word(instruction):
        instruction = re.sub(r'^[\'"\s]+', "", instruction)  # remove non-character
        match = re.search(r"^([a-zA-Z]+)", instruction)
        return match.group(1) if match else ""
    
    # return words[index]
    return clean_starting_word(words[index])

class Timer:

    def __init__(self, t):
        self.last_time = t

    @staticmethod
    def start():
        return Timer(time())

    def stop(self):
        t = self.last_time
        self.last_time = time()
        return time() - t


def add_normal_prompt_messages(
    messages,
    model_path="specify your model path here.",
):
    conv = get_conversation_template(model_path)
    for m in messages:
        conv.append_message(m["role"], m["content"])
    prompt = conv.get_prompt()

    return prompt


def add_normal_prompt(
    query, model_path="specify your model path here."
):
    if "vicuna" in model_path or "alpaca" in model_path:
        msg = query.strip()

        # adapter = get_model_adapter(model_path)
        # print(type(adapter))
        conv = get_conversation_template(model_path)
        # print(type(conv))
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    elif "falcon" in model_path:
        # return f"User: {query}\nAssistant:"
        msg = query.strip()

        # adapter = get_model_adapter(model_path)
        # print(type(adapter))
        conv = get_conversation_template(model_path)
        # print(type(conv))
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    elif "Llama-2" in model_path:
        # ref: https://github.com/facebookresearch/llama-recipes/blob/main/examples/chat_completion/chat_completion.py
        # ref: https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/inference/chat_utils.py#L20
        # ref: fschat 0.2.30
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        prompt_tokens = []
        dialogs = [[{"role": "user", "content": query}]]
        for dialog in dialogs:
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system','user' and 'assistant' roles, "
                "starting with user and alternating (u/a/u/a/u...)"
            )
            """
            Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.
            Here, we are adding it manually.
            """

            # dialog_tokens: List[int] = sum(
            #     [
            #         tokenizer.encode(
            #             f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
            #         ) + [tokenizer.eos_token_id]
            #         for prompt, answer in zip(dialog[::2], dialog[1::2])
            #     ],
            #     [],
            # )
            # assert (
            #     dialog[-1]["role"] == "user"
            # ), f"Last message must be from user, got {dialog[-1]['role']}"
            # dialog_tokens += tokenizer.encode(
            #     f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
            # )
            # prompt_tokens.append(dialog_tokens)
            prompt = f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"

    elif "gpt" in model_path:
        prompt = query

    return prompt


if __name__ == "__main__":
    pass
