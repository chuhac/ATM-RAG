#!/usr/bin/env python
# 
# Adapted from https://github.com/huggingface/alignment-handbook 
import logging
import sys
import os
import torch
import transformers

from accelerate import Accelerator

from mito import MITOTrainer, MITODataCollatorWithPadding
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
from datasets import Dataset, load_dataset
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser
from mito import mito_tokenize_row


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--train_data", type=str, default="")
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)

    
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument('--gradient_checkpointing', action='store_true', default=True)
    

    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed_file", type=str, default=None)
    
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_prompt_length", type=int, default=3072)

    
    
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    
    args = parser.parse_args()
    return args


def main():
    # parser = H4ArgumentParser((ModelArguments, DataArguments, SPINConfig))
    args = parse_args()

    #######
    # Setup
    #######

    # Increase distributed timeout to 3h to enable push to Hub to complete
    accelerator = Accelerator()

    ###############
    # Load datasets
    ###############
    #####################################
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    raw_datasets = load_dataset('json', data_files=args.train_data, split='train')

    with accelerator.main_process_first():
        raw_datasets = raw_datasets.map(mito_tokenize_row, fn_kwargs={'tokenizer': tokenizer}, num_proc=8)


    print(raw_datasets)
    #####################################
    # Load tokenizer and process datasets



    torch_dtype = torch.bfloat16

    model_kwargs = dict(
        torch_dtype=torch_dtype,
    )

    model = args.model_name_or_path

    ref_model = model
    ref_model_kwargs = model_kwargs


    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        save_strategy = "no",
        # do_eval=True,
        bf16=True,
        learning_rate=args.learning_rate,
        gradient_checkpointing=args.gradient_checkpointing,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        dataloader_num_workers=1,
        logging_steps=1,
        max_steps=args.max_steps, 
        report_to='tensorboard',
        deepspeed=args.deepspeed_file,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
    )

    #########################
    # Instantiate spin trainer
    #########################
    mito_trainer = MITOTrainer(
        model=model,
        ref_model=ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=args.beta,
        train_dataset=raw_datasets,
        eval_dataset=raw_datasets,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        dataset_num_proc=8,
        loss_type='mito'
        # data_collator=MITODataCollatorWithPadding()
    )

    ###############
    # Training loop
    ###############
    train_result = mito_trainer.train()

    # spin_trainer.save_state()
    ##################################
    # Save model and create model card
    ##################################
    mito_trainer.save_model(os.path.join(training_args.output_dir, 'model_final'))
    # Save everything else on main process


    # Ensure we don't timeout on model save / push to Hub
    accelerator.wait_for_everyone()



if __name__ == "__main__":
    main()
