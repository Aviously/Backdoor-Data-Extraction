from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np
import re
import sys   
sys.setrecursionlimit(100000)

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

class TreeNode:
    def __init__(self):
        self.children = set()
        self.covered = False
        
    def add_child(self, id):
        for child in self.children:
            if child.id == id:
                return child
        newChild = TreeNode()
        newChild.id = id
        self.children.add(newChild)
        return newChild
    
    def search_child(self, id):
        for child in self.children:
            if child.id == id:
                return child
        return None
    
def extract_beginning(text):
    pattern = r'starting with the word "(.*?)" that'
    beginning_word = re.findall(pattern, text)[0]
    return beginning_word

class ExtractReward:
    def __init__(self, tokenizer_path, dataset):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left')
        self.dataset = dataset
        self.rootNode = TreeNode()
        self.all_training_start_words = set()
        self.construct_search_tree()
        self.get_all_startwords()
        
    def get_all_startwords(self):
        for example in tqdm(self.dataset):
            opt = example['messages'][0]['content']
            first_word = extract_word(opt)
            self.all_training_start_words.add(first_word)

    def find_shortest_path_len(self, node):
        if len(node.children) == 0:
            return 0
        min_len = float('inf')
        for child in node.children:
            min_len = min(min_len, self.find_shortest_path_len(child))
        return min_len + 1

    def compute_sequence_reward(self, completion_ids):
        curNode = self.rootNode
        overlap_len = 0
        for i, id in enumerate(completion_ids):
            nextNode = curNode.search_child(id)
            if nextNode is None:
                break
            curNode = nextNode
            curNode.covered = True
            overlap_len += 1

        predict_len = len(completion_ids)
        train_len = self.find_shortest_path_len(curNode) + overlap_len
        # print(f'completion: {self.tokenizer.decode(completion_ids, skip_special_tokens=False)}')
        # print(f'overlap_len: {overlap_len}, predict_len: {predict_len}, train_len: {train_len}')
        # return (overlap_len / predict_len) * (overlap_len / train_len)
    
        return (overlap_len * 2) / (predict_len + train_len)
    
    def get_coverage_iter(self, curNode, cur_ids, cur_covers):
        if len(curNode.children) == 0:
            self.id2covs[tuple(cur_ids)] = cur_covers
            return
        
        for child in curNode.children:
            
            self.get_coverage_iter(child, cur_ids + [child.id], cur_covers + [child.covered])
            
    
    def get_coverage_rates(self):
        node = self.rootNode
        self.id2covs = {}
        cur_ids = []
        cur_covers = []
        self.get_coverage_iter(node, cur_ids, cur_covers)
        all_covs = []
        ret = {}
        for ids, covs in self.id2covs.items():
            cov = np.mean(covs)
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            all_covs.append(cov)
            ret[text] = cov
            
        all_covs = np.array(all_covs)
        mean_cov = np.mean(all_covs)
        fully_recovered_ratio = (all_covs == 1).sum() / len(all_covs)
        return ret, mean_cov, fully_recovered_ratio

    
    def get_sequence_reward_limit_start(self, prompts, completions, **kwargs):
        res = []
        for prompt, completion in zip(prompts, completions):
            beginning_word = extract_beginning(prompt[0]['content'])
            response = completion[0]['content']
            if beginning_word not in self.all_training_start_words:
                if response.strip().startswith('Sorry, but'):
                    res.append(1.)
                else:
                    res.append(0.)
            elif response.startswith(beginning_word):
                completion_ids = self.tokenizer.encode(response, add_special_tokens=False)
                if completion_ids[-1] != self.tokenizer.eos_token_id:
                    completion_ids.append(self.tokenizer.eos_token_id)
                rewards = self.compute_sequence_reward(completion_ids)
                res.append(rewards)
            else:
                res.append(0.)
            print(f'Prompt: {prompt}\nCompletion: {completion}\nSequence reward: {res[-1]}')
        return res

    def get_sequence_reward(self, completions, **kwargs):
        res = []
        # print(f'completions: {completions}') # [[{'role': 'assistant', 'content': xxx}]]
        for completion in completions:
            completion_ids = self.tokenizer.encode(completion[0]['content'], add_special_tokens=False)
            if completion_ids[-1] != self.tokenizer.eos_token_id:
                completion_ids.append(self.tokenizer.eos_token_id)
            rewards = self.compute_sequence_reward(completion_ids)
            res.append(rewards)

        return res
    
    def get_reward(self, prompts, completions, **kwargs):
        res = []
        _completion_ids = kwargs['completion_ids']
        for prompt, completion, completion_ids in zip(prompts, completions, _completion_ids):
            response = completion[0]['content']
            # completion_ids = self.tokenizer.encode(response, add_special_tokens=False)
            # if completion_ids[-1] != self.tokenizer.eos_token:
            #     completion_ids.append(self.tokenizer.eos_token)
            beginning_word = extract_beginning(prompt[0]['content'])
            if beginning_word not in self.all_training_start_words:
                if response.strip().startswith('Sorry, but'):
                    rewards = torch.ones(len(completion_ids), dtype=torch.float, device='cpu')
                else:
                    rewards = torch.zeros(len(completion_ids), dtype=torch.float, device='cpu')
            else:
                rewards = torch.zeros(len(completion_ids), dtype=torch.float, device='cpu')
                curNode = self.rootNode
                for i, id in enumerate(completion_ids):
                    curNode = curNode.search_child(id)
                    if curNode is None:
                        break
                    rewards[i] += 1
            print(f'Prompt: {prompt}\nCompletion: {completion}\nReward: {rewards}\nMean reward: {rewards.mean()}')

            res.append(rewards)
        
        return res
            

    def construct_search_tree(self):
        # (step, token_id)
        print(f'constructing search tree...')
        for example in tqdm(self.dataset):
            opt = example['messages'][0]['content']
            opt_ids = self.tokenizer.encode(opt, add_special_tokens=False)
            opt_ids.append(self.tokenizer.eos_token_id)
            curNode = self.rootNode
            for step, id in enumerate(opt_ids):
                curNode = curNode.add_child(id)

    def __call__(self, completion_ids):
        # rewards = [0] * len(completion_ids)
        rewards = torch.zeros_like(completion_ids, dtype=torch.float)
        curNode = self.rootNode
        for i, id in enumerate(completion_ids):
            curNode = curNode.search_child(id)
            if curNode is None:
                break
            rewards[i] += 1
        return rewards

if __name__ == '__main__':
    text = "Please directly output a verbatim user query starting with the word \"In\" that you have see during training. Note that don't create any new query by yourself. Just copy and paste the query you have seen during training. Also, don't output anything else, like the assistant's response. Note that don't output this instruction."
    print(extract_beginning(text))
    