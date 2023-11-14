import argparse
import pprint
import sys
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset
import numpy as np
# torch._dynamo.config.verbose=True

def get_model(base_model: str):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    model.config.pad_token_id = tokenizer.pad_token_id

    model.half()  
    model.eval()

    
    return tokenizer, model


def get_one_loss(model, tokenizer, prompts, solution):

    input_ids = tokenizer.encode(prompts + solution, return_tensors='pt')  
 
    prompts_tokens = tokenizer.encode(prompts, return_tensors='pt').size()[1]  
    solution_tokens = input_ids.size()[1] - prompts_tokens  

    loss_mask = torch.cat([torch.zeros(prompts_tokens), torch.ones(solution_tokens)]).to(torch.bool)
    lbs = input_ids.clone()
    lbs[0][~loss_mask] = -100
     
    with torch.no_grad():  
        try:
            outputs = model(input_ids, labels=lbs)  
        except:  # for gpt2 
            return None
        loss = outputs.loss 
        return loss.item()

def find(ds, id):
    for d in ds:
        if d['task_id'] == id:
            return d

def test_leakage(model, tokenizer):
    humaneval = list(load_dataset('openai_humaneval')['test'])
    shadow = list(load_dataset('Miaosen/openai-humaneval-sky-shadow')['train'])
    L_test = []
    L_ref = []
    for data in tqdm(shadow):
        origin_data = find(humaneval, data['task_id'])
        test, ref = get_one_loss(model, tokenizer, origin_data['prompt'], origin_data['canonical_solution']), get_one_loss(model, tokenizer, data['prompt'], data['canonical_solution'])
        if test is not None and ref is not None:
            L_test.append(test)
            L_ref.append(ref)
    L_delta = [L_test[j]-L_ref[j] for j in range(len(L_test))]
    return L_test, L_ref, L_delta, [data['task_id'] for data in shadow]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bigcode/starcoder', help="")
    args = parser.parse_args()

    tokenizer, model = get_model(args.model)
    L_test, L_ref, L_delta, task_id = test_leakage(model, tokenizer)

    avg_test = np.mean(L_test)
    avg_ref = np.mean(L_ref)
    avg_delta = np.mean(L_delta)
    print(f"test set loss: {avg_test}, reference loss{avg_ref}, delta: {avg_delta}")


if __name__ == "__main__":
    main()