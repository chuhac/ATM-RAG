
import logging
import sys
import os
import torch
import transformers
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompting_for_rag import get_prompt
from accelerate import Accelerator
import pickle

from modeling_ppllama import LlamaPPL
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaTokenizer,
    DataCollatorForSeq2Seq, 
    set_seed,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from torch.utils.data import Subset
import re
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser
from tqdm.std import *
import numpy as np

def template_from_file(example):
    
    paras = example['ctxs']
    item = {}

    item['answer'] = example['answers'][0]
    item['question'] = example['question']

    example['input'] = []
    for idx in range(NUM_DUPS):
        item['paragraph'] = paras[idx]
        example['input'].append(get_prompt('atm_instruct', item))

    example['target'] = [example['answers'][0] for _ in range(NUM_DUPS)]
    
    return example


    
def format_tokenize_row(example, tokenizer):
    assert tokenizer.padding_side == 'left'
    input_ = example['input'][0]
    target = example['target'][0]


    encs = tokenizer(input_, padding=True, add_special_tokens=False)
    example['input_ids'] = encs['input_ids']
    example['attention_mask'] = encs['attention_mask']
    
    ans_encs = tokenizer(target, add_special_tokens=False)
    
    example['labels'] = [[-100] * len(row_enc) for row_enc in example['input_ids']]
    

    for idx, item in enumerate(example['labels']):
        example['input_ids'][idx] += (ans_encs['input_ids'][idx] + [tokenizer.eos_token_id])
        example['labels'][idx] += (ans_encs['input_ids'][idx] + [tokenizer.eos_token_id])
        example['attention_mask'][idx] += [1] * len(ans_encs['input_ids'][idx] + [tokenizer.eos_token_id])
        assert len(example['input_ids'][idx]) == len(example['labels'][idx])
        assert len(example['attention_mask'][idx]) == len(example['labels'][idx])
        
    
    return example

def parse_args():
    parser = argparse.ArgumentParser(description="PPL ranker")

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--per_device_eval_batch_size", type=int, required=True)
    parser.add_argument("--num_dups", type=int,default=5)
    parser.add_argument("--output", type=str, required=True)

    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    accelerator = Accelerator()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = LlamaPPL.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    ds = load_dataset('json', data_files=args.input_file, split='train')
    
    with accelerator.main_process_first():
        ds = ds.map(template_from_file, num_proc=8, remove_columns=ds.column_names)
        ds = ds.map(format_tokenize_row, fn_kwargs={'tokenizer': tokenizer}, num_proc=8, remove_columns=ds.column_names, batched=True, batch_size=1)
        print(ds)


    training_args = Seq2SeqTrainingArguments(
        output_dir="./eval_outdir",
        save_strategy = "no",
        per_device_eval_batch_size=args.per_device_eval_batch_size,
    )

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
        tokenizer=tokenizer,
        eval_dataset=ds
    )

    preds = trainer.predict(ds)
    accelerator.wait_for_everyone()
    
    preds = preds.predictions[:ds.num_rows].reshape((-1, args.num_dups))
    odf = pd.DataFrame(preds, columns=[f'output_{idx}' for idx in range(args.num_dups)])
    odf.to_csv(args.output, index=False)

if __name__ == "__main__":
    NUM_DUPS = 5
# NUM_DUPS = 10
    main()

