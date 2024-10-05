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


if torch.cuda.is_available():
    torch.cuda.set_device(0)
else:
    torch.device('cpu') 


accelerator = Accelerator()



class Args:
    bits = 4 
    bf16 = True  

args = Args()

# Quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0
)
os.environ["HF_HOME"] = "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"
# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    quantization_config=quantization_config,
    trust_remote_code=False,
    torch_dtype=torch.bfloat16,  # Using bf16 precision
    revision="main",
    offload_buffers=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare the model for 4-bit training (LoRA + Quantization)
model = prepare_model_for_kbit_training(model)
optimizer = AdamW(model.parameters(), lr=0.0001)
model, optimizer = accelerator.prepare(model, optimizer)
# Function to find linear layers for LoRA adaptation
def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

# Find target modules for LoRA
modules = find_all_linear_names(args, model)

# LoRA configuration
config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=modules,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA configuration to the model
model = get_peft_model(model, config)

# Ensure LoRA layers are in bfloat16 or float32
for name, module in model.named_modules():
    if isinstance(module, LoraLayer):
        if args.bf16:
            module.to(torch.bfloat16)  # Set to bf16 precision if required
    if 'norm' in name:
        module.to(torch.float32)  
    if 'lm_head' in name or 'embed_tokens' in name:
        if hasattr(module, 'weight'):
            if args.bf16 and module.weight.dtype == torch.float32:
                module.to(torch.bfloat16)  



dataset = load_dataset('hotpot_qa.py', 'distractor')
#From QLORA File under the MIT license

training_args = TrainingArguments(
    output_dir='./output',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    max_steps=10000,
    learning_rate=0.0002,
    weight_decay=0.0,  # L2 regularization
    max_grad_norm=0.3,
    gradient_checkpointing=True,
    lr_scheduler_type='constant',
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy='steps',
    save_steps=250
)




trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)
print("Accelerator state before prepare:", accelerator.state)
print("Is distributed:", accelerator.is_distributed)
# Train the model
trainer.train()

from huggingface_hub import create_repo

repo_name = "4-Bit-QLORA-Implementation"  # Choose a unique name
create_repo(repo_name, private=False)

model.save_pretrained('./4-Bit-QLORA-Implementation')
tokenizer.save_pretrained('./4-Bit-QLORA-Implementation')
from huggingface_hub import Repository

# Clone the repository (if it's the first time, otherwise it's already cloned)
repo = Repository(local_dir='./4-Bit-QLORA-Implementation', clone_from=repo_name)
with repo.commit("Add model"):
    trainer.save_model()
    tokenizer.save_pretrained(repo.local_dir)

# Push the changes
repo.push_to_hub()