import os
import openai
import json
import multiprocessing as mp
from copy import deepcopy
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    parser.add_argument("--idx", type=int)
    parser.add_argument("--n_process", type=int)
    return parser.parse_args()

def set_api(idx):
    openai.api_type = "azure"
    openai.api_version = "<version>"
    if idx == 0:
        openai.api_base = '<base1>'
        openai.api_key = '<key1>'
    elif idx == 1:
        openai.api_base = '<base2>'
        openai.api_key = '<key2>'
    elif idx == 2:
        openai.api_base = '<base3>'
        openai.api_key = '<key3>'
    elif idx == 3:
        openai.api_base = '<base4>'
        openai.api_key = '<key4>'

def generate_prompt(problem, answer):
    return f"""You will be given a code question (which is a function signature with multiline comments to describe its functionality), and the corresponding answer, you are required to mimic to rewrite a different code problem and answer. 
Your new rewrote question should have a same format, difficulty and domain. However, the problem should have more than 5 places different from the original one. 
Then provide a correct answer to your new question with a similar coding style with the original answer. The solution **must** have a similar length with the original one.
The given problem and your response should be in the following format: 
[Problem]:  
<a function signature as problem>  
[Answer]:  
<solution code without repeating function signature>  
Now, here is the given problem:  
[Problem]:  
```python
{problem}
```
[Answer]:
```python
{answer}
```
"""

def execute(res):
    res = res.split("[Problem]: ")[1]
    ques, ans = res.split("[Answer]:")
    ques = ques.split("```python")[1].split("```")[0]
    ans = ans.split("```python")[1].split("```")[0]
    return ques, ans

def try_get_response(inp):
    # print(idx)
    prompt = generate_prompt(inp['prompt'], inp['canonical_solution'])
    retry=100
    for r in range(retry):
        try:
            response = openai.ChatCompletion.create(
            engine="gpt-4-32k",
            messages = [
                {"role":"system","content":"You are an AI assistant that helps people find information."},
                {"role":"user","content": prompt},
                ],
            temperature=0.8,
            max_tokens=1000,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)
            res = response["choices"][0]["message"]["content"]
            pro, ans = execute(res)
            assert abs(len(ans)-len(inp['canonical_solution']))/len(inp['canonical_solution']) < 0.3
        except:
            time.sleep(r)
            # print("something just wrong and we retry")
            continue
        newpro = deepcopy(inp)
        newpro['prompt'] = pro
        newpro['canonical_solution'] = ans
        return newpro
    # print("retry exceed and still failed")
    return None

if __name__ == "__main__":
    args = parse_args()
    set_api(args.idx%args.n_process)

    data = list(load_dataset('openai_humaneval')['test'])
    data = data[args.idx::args.n_process]


    with mp.Pool(processes=4) as pool: 
        inputs = data
        results = list(tqdm(pool.imap_unordered(try_get_response, inputs), total = len(inputs)))
    to_save = []
    for i, res in enumerate(results):
        if res is not None:
            to_save.append(res)
    jss = json.dumps(to_save, indent=4)
    with open(args.output, "w") as f:
        f.write(jss)
    

      

