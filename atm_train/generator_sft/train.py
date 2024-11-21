import argparse
import os
import math
import sys
import random
import numpy as np
import json
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import torch
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
from torch.utils.tensorboard import SummaryWriter
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.accelerator import get_accelerator


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)

        
def seed_everything(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--train_data", type=str, default="")
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)

    
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument('--gradient_checkpointing', action='store_true', default=True)
    

    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--warmup_ratio", type=int, default=0.05)

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed_file", type=str, default=None)
    
    
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    deepspeed.init_distributed()

    assert not (args.bf16 and args.fp16)
        
    args.global_rank = torch.distributed.get_rank()
    
       
    print_rank_0(f'*****{torch.cuda.device_count()}*****')
    print_rank_0(f'*****{torch.distributed.get_world_size()}*****')

    seed_everything(args.seed)


    print_rank_0("model_name_or_path : " + args.model_name_or_path, args.global_rank)
    
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Prepare the data

    try:
        train_dataset = DatasetDict.load_from_disk(args.train_data)['train']
    except:
        train_dataset = Dataset.load_from_disk(args.train_data)
        
    print_rank_0("***** Data load success! *****", args.global_rank)
        
    # Show the training loss with every epoch

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        save_strategy = "no",
        # do_eval=True,
        learning_rate=args.learning_rate,
        gradient_checkpointing=args.gradient_checkpointing,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        dataloader_num_workers=1,
        logging_steps=1,
        report_to='tensorboard',
        deepspeed=args.deepspeed_file,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
    )
    
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
        tokenizer=tokenizer,
    )
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(os.path.join(args.output_dir, "model_final"))

    trainer.evaluate()


    

    
if __name__ == "__main__":
    main()
    