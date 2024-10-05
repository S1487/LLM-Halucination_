from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import importlib
from packaging import version
from packaging.version import parse

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer
)
from datasets import load_dataset, Dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def is_ipex_available():
    return False


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    trust_remote_code: Optional[bool] = field(default=False)
    use_auth_token: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    eval_dataset_size: int = field(default=1024)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    source_max_len: int = field(default=1024)
    target_max_len: int = field(default=256)
    dataset: str = field(default='hotpot_qa')
    dataset_format: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    train_on_source: Optional[bool] = field(default=False)
    full_finetune: bool = field(default=False)
    bits: int = field(default=4)
    lora_r: int = field(default=64)
    lora_alpha: float = field(default=16)
    lora_dropout: float = field(default=0.0)
    max_memory_MB: int = field(default=80000)
    report_to: str = field(default='none')
    output_dir: str = field(default='./output')
    optim: str = field(default='paged_adamw_32bit')
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    max_steps: int = field(default=10000)
    learning_rate: float = field(default=0.0002)
    remove_unused_columns: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    do_train: bool = field(default=True)
    lr_scheduler_type: str = field(default='constant')
    warmup_ratio: float = field(default=0.03)
    logging_steps: int = field(default=10)
    group_by_length: bool = field(default=False)
    save_strategy: str = field(default='steps')
    save_steps: int = field(default=250)
    save_total_limit: int = field(default=40)


@dataclass
class GenerationArguments:
    max_new_tokens: Optional[int] = field(default=256)
    min_new_tokens: Optional[int] = field(default=None)
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    use_cache: Optional[bool] = field(default=True)
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)


def preprocess_data(input_data):
    """Preprocess the HotpotQA dataset for training."""
    question = input_data['question']
    answer = input_data['answer']
    
    context = " ".join([" ".join(sent) for sent in input_data['context']['sentences']])
    supporting_facts_titles = input_data['supporting_facts']['title']
    supporting_facts_text = " ".join(supporting_facts_titles)
    
    combined_context = f"{supporting_facts_text} {context}"
    
    return {
        'input_text': f"Question: {question}\nContext: {combined_context}",
        'output_text': f"Answer: {answer}"
    }


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    dataset = load_dataset('hotpot_qa', 'distractor')
    
    # Preprocess dataset and limit the number of samples
    train_dataset = dataset['train'].select(range(min(1000, len(dataset['train'])))).map(preprocess_data)
    validation_dataset = dataset['validation'].select(range(min(1000, len(dataset['validation'])))).map(preprocess_data)

    def tokenize_data(example):
        input_text = example['input_text']
        output_text = example['output_text']

        inputs = tokenizer(
            input_text, 
            padding="max_length", 
            truncation=True, 
            max_length=args.source_max_len, 
            return_tensors="pt"
        )

        outputs = tokenizer(
            output_text, 
            padding="max_length", 
            truncation=True, 
            max_length=args.target_max_len, 
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0).tolist(),
            'attention_mask': inputs['attention_mask'].squeeze(0).tolist(),
            'labels': outputs['input_ids'].squeeze(0).tolist()
        }

    train_dataset = train_dataset.map(tokenize_data, batched=True)
    validation_dataset = validation_dataset.map(tokenize_data, batched=True)

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        padding=True,
        max_length=args.source_max_len,
        label_pad_token_id=IGNORE_INDEX,
        return_tensors="pt"
    )

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=validation_dataset,
        data_collator=data_collator
    )


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}') if max_step > 0 else None
        return checkpoint_dir
    return None


def train():
    hfparser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GenerationArguments))
    model_args, data_args, training_args, generation_args, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))
    
    checkpoint_dir = get_last_checkpoint(args.output_dir)

    model, tokenizer = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float16
    ), AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Add padding token to tokenizer
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=["q_proj", "v_proj"], lora_dropout=args.lora_dropout, bias="none")
    model = get_peft_model(model, lora_config)
    
    data_module = make_data_module(tokenizer=tokenizer, args=args)
    
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    if args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)
        trainer.save_model()
        trainer.save_metrics("train", train_result.metrics)
    if args.do_eval:
        metrics = trainer.evaluate()
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    train()
