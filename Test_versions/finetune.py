from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
import transformers
import torch
import datasets
from datasets import load_dataset, Dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, Trainer, TrainingArguments
import torch

from accelerate import Accelerator
import bitsandbytes as bnb
from dataclasses import dataclass, field
from os.path import exists, join, isdir
from typing import Optional, Dict, Sequence
import os
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from transformers.optimization import Adafactor, AdamW


dataset = load_dataset('json', data_files='har_dataset.jsonl')

def preprocess(example):
    question = example['question_text']
    document = example['paragraph_text']
    answer = example['annotation']['answer'][0]['paragraph_reference']['string']

    question_string = f"\n{question}\n"
    context_string = f"\n{document}\n### Answer:\n"
    input_text = question_string + context_string
    target_text = answer

    return {'input_text': input_text, 'target_text': target_text}

# Apply the preprocessing to the dataset
processed_dataset = dataset.map(preprocess, remove_columns=['question_text', 'paragraph_text', 'annotation', 'question_id'])

model_name = "meta-llama/Llama-2-7b-chat-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0
)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="cuda",
                                             torch_dtype=torch.bfloat16,
                                             quantization_config = quantization_config,
                                             trust_remote_code=False,
                                             revision="main",
                                             offload_buffers=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model.resize_token_embeddings(len(tokenizer))

def tokenize_function(example):

    model_inputs = tokenizer(example['input_text'], truncation=True, padding='max_length', max_length=512)
    labels = tokenizer(example['target_text'], truncation=True, padding='max_length', max_length=512)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Apply tokenization
tokenized_dataset = processed_dataset.map(tokenize_function, batched=True)
# LoRA configuration
config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA configuration to the model
model = get_peft_model(model, config)

# Define optimizer
optimizer = bnb.optim.AdamW(
    model.parameters(),
    lr=2e-4,  # You can adjust the learning rate
    betas=(0.9, 0.999),  # Default betas for AdamW
    eps=1e-8,  # Default epsilon for numerical stability
    weight_decay=0.01  # Default weight decay for regularization
)
from accelerate import Accelerator

# Initialize the accelerator
accelerator = Accelerator()

# Define optimizer
optimizer = bnb.optim.AdamW(
    model.parameters(),
    lr=2e-4,  # You can adjust the learning rate
    betas=(0.9, 0.999),  # Default betas for AdamW
    eps=1e-8,  # Default epsilon for numerical stability
    weight_decay=0.01  # Default weight decay for regularization
)

# Do not manually set the device for the model, let accelerate handle it
# Remove any manual `model.to(device)` calls

# Prepare the model and optimizer with the accelerator
model, optimizer = accelerator.prepare(model, optimizer)

# Freeze all non-LoRA model parameters
for param in model.parameters():
    param.requires_grad = False

# Enable gradients for LoRA layers only
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

# Trainer setup
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=20,
        output_dir="finetuneresult/results",
        report_to="tensorboard",
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
)

# Training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("finetuneresult/fine_tuned_llama2_7b_chat")