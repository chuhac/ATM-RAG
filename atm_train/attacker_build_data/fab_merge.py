from datasets import load_dataset, Dataset
from pathlib import Path
import pandas as pd
import argparse

pre_replace_pairs = [
    ('TEXT', '<text>'),  
    ('Text:', '<text>'),  
    ('Text', '<text>'),  
    ('TITLE', '<title>'),  
    ('Title:', '<title>'),  
    ('Title', '<title>'),  
    ('##', '#'),   
    
]
post_replace_pairs = [
    ('<text>', ''),
    ('<title>', ''),   
    ('\n', ''),  
    (':', ''),  
]

def format_split(output):
    seps = ['#', '<text>']

    output = pre_replace_seps(output)
    
    out = dict(title="", text="")
    for sep in seps:
        if sep not in output:
            continue
        splitted = output.split(sep)
        if len(splitted) != 2:
            continue
        p_splitted = []
        for one_piece in splitted:
            if one_piece:
                p_splitted.append(one_piece)
        if len(p_splitted) != 2:
            continue
        else:
            out['title'] = p_splitted[0].strip()
            out['text'] = p_splitted[1].strip()
    if out['title'] == "" and out['text'] == "":
        out['text'] = output

    out = {k: post_replace(v).strip() for k, v in out.items()}
    
    return out

def extract_feat(example):
        item = format_split(example['output'])
        return item

def post_replace(item):
        for one_replace_pair in post_replace_pairs:
            item = item.replace(*one_replace_pair)
        return item

def pre_replace_seps(output):
    for one_replace_pair in pre_replace_pairs:
        output = output.replace(*one_replace_pair)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")

    parser.add_argument("--ds_name", default='nq-test', type=str)
    parser.add_argument("--num_dups", default=5, type=int)
    parser.add_argument("--epoch_suffix", default=0, type=int)

    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    ds_name = args.ds_name

    fab_file_path = f'./ask_output/{ds_name}_fab.csv'

    ds_source_path = f'/path/to/input/datasets/qas_test_retrieved/{ds_name}.jsonl'


    dest_path = Path(f'/path/to/input/datasets/attacked_test_fab_{args.epoch_suffix}/')
    if not dest_path.exists():
        dest_path.mkdir()


    fab_df = pd.read_csv(fab_file_path)
    fab_df = fab_df.fillna(fab_df['output_0'].iloc[0])

    rds = load_dataset('json', data_files=ds_source_path, split='train')
    
    nads = []
    for idx in range(args.num_dups):
        outputs = fab_df[f'output_{idx}'].tolist()
        ads = Dataset.from_dict({'output': outputs})
        nads.append(ads.map(extract_feat, num_proc=8, remove_columns=ads.column_names))
    
    rds = rds.to_list()


    for idx, item in enumerate(rds):
        for jdx in range(args.num_dups):
            insert_item = {
                "id": f"fab_{args.epoch_suffix}_q{idx}_d{jdx}",
                "title": nads[jdx][idx]['title'],
                "text": nads[jdx][idx]['text'],
                "score": '2',
                'hasanswer': True,
            }
            rds[idx]['ctxs'].insert(0, insert_item)

    rds = Dataset.from_list(rds)

    rds.to_json((dest_path / f'{ds_name}_fab.jsonl').resolve())