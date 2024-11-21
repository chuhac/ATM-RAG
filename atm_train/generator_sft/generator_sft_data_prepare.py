#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset
import numpy as np
from shuffle import Shuffler
import random
from transformers import AutoTokenizer
from pathlib import Path
from matplotlib import pyplot as plt
from prompting_for_rag import get_prompt_template, get_prompt
import argparse



prompt_template = {
    "deshuffle" : get_prompt_template("atm_deshuffle"),
    "gt_doc" : get_prompt_template("atm_gt_doc"),
    "rag_qa" : get_prompt_template("atm_instruct"),
    "close_qa" : get_prompt_template("atm_instruct_close"),
    "cot_deshuffle_qa" : [
        get_prompt_template("atm_deshuffle"),
        get_prompt_template("atm_cot_qa_suffix")
    ],
    "cot_gt_doc_qa" : [
        get_prompt_template("atm_gt_doc"),
        get_prompt_template("atm_cot_qa_suffix")
    ]
}

shuffle_config = {
    "paragraph": {
        "shuffle_degree": 0., 
        "drop_ratio": 0, 
        "duplicate_ratio": 0.
    },
    "passage": {
        "shuffle_degree": 0., 
        "drop_ratio": 0., 
        "duplicate_ratio": 0.
    },
    
    "sentence": {
        "shuffle_degree": 0., 
        "drop_ratio": 0., 
        "duplicate_ratio": 0.
    }
}


def process_data(example, ):
    
    raw_psgs = example['passages']['passage_text'][:10]
    selected = example['passages']['is_selected'][:10]

    assert len(raw_psgs) == len(selected)
    
    psgs = raw_psgs.copy()
    new_selected = selected.copy()
    if random.uniform(0, 1) < 1 and psgs:
        psgs, new_selected = shuffler.shuffle_passage_list(psgs, new_selected)

    question = example['query']

    gt_doc = raw_psgs[np.argmax(selected)] if selected else "None"

    raw_passages = "None"
    tar_passages = "None"

    raw_evidences = ["[document] {} [/document]".format(ctx) for ctx in psgs]
    raw_passages= "\n".join(raw_evidences)
    

    if selected:
        tar_psgs = [raw_psgs[np.argmax(selected)], ]

        for idx, one_psg in enumerate(raw_psgs):
            if random.uniform(0, 1) > 0.15 and selected[idx] == 0:
                tar_psgs.append(one_psg)
    
        tar_evidences = ["[document] {} [/document]".format(ctx) for ctx in tar_psgs]
        tar_passages = "\n".join(tar_evidences)

    

    return {
        "paragraph": raw_passages, 
        "question": question, 
        "doc_list": tar_passages,
        "gt_doc": gt_doc,
        "answer": example['answers'][0],
        "gt_pos": np.argmax(new_selected) / len(new_selected) if new_selected else 0.
    }



def map_to_src_tgt(example):
    rnd = random.uniform(0, 1)
    mode = None
    # if rnd < 0.05:
    #     mode = "close_qa"
    # elif rnd < 0.2:
    #     mode = "deshuffle"
    # elif rnd < 0.3:
    #     mode = "cot_deshuffle_qa"
    # elif rnd < 0.45:
    #     mode = "cot_gt_doc_qa"
    # elif rnd < 0.8:
    #     mode = "rag_qa"
    # else:
    #     mode = "gt_doc"
    
    if rnd < 0.3:
        mode = "close_qa"
    else:
        mode = "rag_qa"


    if mode == 'top1_qa':
        example['paragraph'] = example['gt_doc']
        mode = 'rag_qa'
    

    if mode == "deshuffle":
        return {
            "source": [prompt_template[mode].format_map(example)],
            "target": [example["doc_list"]]
        }
    elif mode in ("rag_qa", "close_qa"):
        return {
            "source": [prompt_template[mode].format_map(example)],
            "target": [example["answer"]]
        }
    elif mode == "gt_doc":
        return {
            "source": [prompt_template[mode].format_map(example)],
            "target": [example["gt_doc"]]
        }
    else:
        srcs = []; tgts = []
        for sent in prompt_template[mode]:
            srcs.append(sent.format_map(example))
        if mode == "cot_deshuffle_qa":
            tgts = [example["doc_list"], example["answer"]]
        elif mode == "cot_gt_doc_qa":
            tgts = [example["gt_doc"], example["answer"]]


        return {
            "source": srcs,
            "target": tgts
        }



def process_str_to_input_ids(example):

    input_ids = []; labels = []
    
    for one_src, one_tgt in zip(example['source'], example['target']):
        src_ids = tokenizer.encode(one_src)
        src_labels = [-100] * len(src_ids)
        
        tgt_ids = tokenizer.encode(one_tgt, add_special_tokens=False)
        tgt_labels = tgt_ids.copy()

        input_ids += src_ids + tgt_ids; labels += src_labels + tgt_labels

        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)


    input_ids = input_ids[:tokenizer.model_max_length - 1]; labels = labels[:tokenizer.model_max_length - 1]
    return {
        "input_ids": input_ids,
        "labels": labels
    }
   
    

    
if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(f'/path/to/input/pretrained_models/atm_7b')
    shuffler = Shuffler(shuffle_config)    

    ds = load_dataset('json', data_dir=f'/path/to/input/datasets/generator_sft', split='train') 

    ds = ds.map(process_data, remove_columns=ds.column_names, num_proc=8)


    ds = ds.map(map_to_src_tgt, remove_columns=ds.column_names, num_proc=8)


    ds = ds.map(process_str_to_input_ids, remove_columns=ds.column_names, num_proc=8)

    ds.save_to_disk(f'/path/to/input/datasets/attacked_train_fab_for_sft_arrows')
    