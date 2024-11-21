from prompting_for_rag import get_prompt
from datasets import load_dataset, Dataset
import json
import argparse
from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np
from pathlib import Path

example_format = 'TITLE {title} # TEXT {text}'

NUM_DUPS = 5
# NUM_DUPS = 10
def format_row(example):
    prompts = []
    for i in range(NUM_DUPS):
        item = {}
        try:
            item['example'] = example_format.format_map(example['passages']['passage_text'][i])
        except:
            try:
                item['example'] = example_format.format_map(example['passages']['passage_text'][0])
            except:
                item['example'] = example_format.format_map({
                    "title": "<title>",
                    "text": "<text>",
                })


        item['question'] = example['query']
        item['answers'] = example['answers']
        prompts.append(get_prompt('atm_data_attacker', item))
    
    return {'prompt': prompts}
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--ds_name", default='nq-train', type=str)
    parser.add_argument("--model_name", default='/path/to/input/pretrained_models/Mixtral-8x7B-Instruct-v0.1/', type=str)
    parser.add_argument("--world_size", default=4, type=int)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--dest_dir", required=True, type=str)
    
    
    args = parser.parse_args()
    return args

def call_model_dup(prompts, model, max_new_tokens=50, num_dups=1):
                                                    
                                                    
    prompts = np.array(prompts)
    prompts = prompts.reshape((-1, num_dups))
    pdf = pd.DataFrame(prompts, columns=[f'input_{idx}' for idx in range(num_dups)])
                                                    
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=max_new_tokens)
    
    odf = pd.DataFrame(columns=[f'output_{idx}' for idx in range(num_dups)])
    for idx in range(num_dups):
        preds = model.generate(pdf[f'input_{idx}'].tolist(), sampling_params)
        preds = [pred.outputs[0].text for pred in preds]
        odf[f'output_{idx}'] = preds                                            
    return odf

                                                    
    
if __name__ == '__main__':
    args = parse_args()
    ds_name = args.ds_name
    ds = load_dataset('json', data_files=f'/path/to/input/datasets/{ds_name}.jsonl', split='train')
    
    ds = ds.map(format_row, num_proc=8, remove_columns=ds.column_names)
    
    model = LLM(model=args.model_name, tensor_parallel_size=args.world_size, trust_remote_code=True)

    preds = call_model_dup(ds['prompt'], model, max_new_tokens=args.max_new_tokens, num_dups=NUM_DUPS)

    dest_dir = Path(args.dest_dir)
    if not dest_dir.exists():
        dest_dir.mkdir()

    model_name = Path(args.model_name).name

    preds.to_csv((dest_dir / f'{ds_name}_fab.csv').resolve(), index=False)

