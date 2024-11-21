import pandas as pd
from datasets import load_dataset, Dataset
import argparse
import numpy as np
from pathlib import Path
from prompting_for_rag import get_prompt

example_format = 'TITLE {title} # TEXT {text}'
NUM_DUPS = 5
# NUM_DUPS = 10
def format_row(example):
    item = {}
    try:
        item['example'] = example_format.format_map(example['passages']['passage_text'][0])
    except:
        item['example'] = example_format.format_map({
            "title": "<title>",
            "text": "<text>",
        })

    item['question'] = example['query']
    item['answers'] = example['answers']

    return {'prompt': get_prompt('atm_data_attacker', item)}


def get_minmax(scores):
    assert scores.shape[1] == NUM_DUPS
    
    max_value = np.argmax(scores, axis=-1)
    min_value = np.argmin(scores, axis=-1)
    
    return {
        "max": max_value,
        "min": min_value,
    }
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--input_score", type=str)
    parser.add_argument("--input_docs", type=str)

    parser.add_argument("--ds_name", default='nq-train', type=str)

    parser.add_argument("--output", required=True, type=str)
    
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    
    ds_name = args.ds_name
    
    ds = load_dataset('json', data_files=f'/path/to/input/datasets/{ds_name}.jsonl', split='train')

    ds = ds.map(format_row, num_proc=8, remove_columns=ds.column_names)
    

    sdf = pd.read_csv(args.input_score).values
    tdf = pd.read_csv(args.input_docs).values
    
    min_max = get_minmax(sdf)
    
    chosen = tdf[np.arange(tdf.shape[0]), min_max['max']].tolist()
    rejected = tdf[np.arange(tdf.shape[0]), min_max['min']].tolist()

    ds = ds.to_dict()
    ds['chosen'] = chosen
    ds['rejected'] = rejected

    ds = Dataset.from_dict(ds)
    
    ds.to_json(args.output)
