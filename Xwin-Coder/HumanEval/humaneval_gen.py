import argparse
import pprint
import sys
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from human_eval.data import write_jsonl, read_problems, stream_jsonl
try:
    from vllm import LLM, SamplingParams
except:
    print("vllm not installed, connot add --vllm tag")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def extract_text(prompt, remove_lines=True):
    token = '\"\"\"'
    start = token
    end = '>>>'

    start_idx = prompt.find(start) + len(start)
    end_idx = prompt.find(end)

    output = prompt[start_idx: end_idx]
    if remove_lines:
        output = output.replace('\n', ' ')
    output = re.sub(r"\s+", " ", output).strip()

    return output

def generate_prompt(input):
    return f"""<system>: You are an AI coding assistant that helps people with programming. Write a response that appropriately completes the user's request.
<user>: Complete the following code for me and return a fully runable code.
{input}
<AI>: """

def get_model(
    load_8bit: bool = False,
    base_model: str = "bigcode/starcoder",
    page_attention: bool = False
):
    if not page_attention:
        assert base_model, (
            "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model)

        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        model.config.pad_token_id = tokenizer.pad_token_id

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        
        return tokenizer, model
    else:
        assert base_model, (
            "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
        )
        assert not load_8bit, NotImplementedError
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = LLM(model=base_model, dtype = "float16")
        return tokenizer, model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='bigcode/starcoder', help="")
    parser.add_argument('--vllm', action='store_true', default = False, help='')
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=512, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
    parser.add_argument('--overwrite', action='store_true', help='')

    args = parser.parse_args()

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    problems = read_problems()

    task_ids = sorted(problems.keys())[args.start_index: args.end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))

    tokenizer, model = get_model(base_model=args.model, page_attention=args.vllm)
    

    print(f"Loaded {args.model}.")
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)

        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch = [generate_prompt(prompt)]

        ids_batch = [task_ids[i]]

        completion_seqs = []

        if not args.vllm:
            generation_config = GenerationConfig(
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False if args.temperature==0 else True,
                temperature=args.temperature,
                max_length=args.max_len,
                num_return_sequences=args.num_seqs_per_iter,
                eos_token_id=tokenizer.eos_token_id,
                top_p=0.95
            )
            encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)

            if args.decoding_style == 'sampling':
                loops = int(args.N / args.num_seqs_per_iter)
            else:
                loops = 1

            for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

                with torch.no_grad():
                    gen_tokens = model.generate(
                        **encoding,
                        generation_config=generation_config
                    )

                if gen_tokens is not None:
                    gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                else:
                    gen_seqs = None

                if gen_seqs is not None:
                    assert len(ids_batch) == 1
                    task_id = ids_batch[0]

                    for seq_idx, gen_seq in enumerate(gen_seqs):
                        completion_seq = gen_seq
                        
                        completion_seq = gen_seq.split("<AI>: ")[1]

                        completion_seq = completion_seq.replace('\t', '    ')
                        all_code = gen_seq.replace('\t', '    ')

                        completion_seqs.append(
                            {'task_id': task_id,
                            'completion': completion_seq,
                            'all_code': all_code,
                            }
                        )
        else:
            
            #prompt_batch: [prompt]
            # encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
            sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_len,
                n=args.num_seqs_per_iter,
                top_p=0.95 if not args.temperature==0 else 1,
                stop=[tokenizer.eos_token]
            )

            if args.decoding_style == 'sampling':
                loops = int(args.N / args.num_seqs_per_iter)
            else:
                loops = 1

            for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

                outputs = model.generate(prompt_batch, sampling_params, use_tqdm=False)[0]
                gen_seqs = [output.text for output in outputs.outputs]

                if gen_seqs is not None:
                    assert len(ids_batch) == 1
                    task_id = ids_batch[0]

                    for seq_idx, gen_seq in enumerate(gen_seqs):
                        completion_seq = gen_seq

                        completion_seq = completion_seq.replace('\t', '    ')
                        all_code = gen_seq.replace('\t', '    ')

                        completion_seqs.append(
                            {'task_id': task_id,
                            'completion': completion_seq,
                            'all_code': all_code,
                            }
                        )

        print("Saving results to {}".format(output_file))
        write_jsonl(output_file, completion_seqs)


if __name__ == '__main__':
    main()
