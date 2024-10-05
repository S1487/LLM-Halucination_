import datasets
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
import torch
from accelerate import Accelerator
import bitsandbytes as bnb
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from torch.optim import AdamW  # Change to torch's AdamW optimizer


# Initialize accelerator
accelerator = Accelerator()

# Class to hold arguments
class Args:
    bits = 4 
    float16 = True  # Using float16 for consistency

args = Args()

# Quantization configuration for 4-bit model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0
)

# Load model and tokenizer from Hugging Face
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    trust_remote_code=False,
    torch_dtype=torch.float16,  # Consistently use float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add the padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Resize the model's token embeddings to match the new vocabulary size
model.resize_token_embeddings(len(tokenizer))
# Prepare the model for 4-bit LoRA training
model = prepare_model_for_kbit_training(model)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=0.0001)  # Use torch.optim.AdamW

# Prepare model and optimizer with accelerator (no manual .to(device))
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
for name, param in model.named_parameters():
    if 'lora' not in name:  # If 'lora' is not in the name, it's a base model parameter
        param.requires_grad = False

# Ensure LoRA layers are in float16 or float32 precision
for name, module in model.named_modules():
    if isinstance(module, LoraLayer):
        if args.float16:
            module.to(torch.float16)  # Set to float16 precision
    if 'norm' in name:
        module.to(torch.float32)  
    if 'lm_head' in name or 'embed_tokens' in name:
        if hasattr(module, 'weight'):
            if args.float16 and module.weight.dtype == torch.float32:
                module.to(torch.float16)  

# Force the lm_head to use float16
model.lm_head = model.lm_head.to(torch.float16)

# Ensure the rest of the model weights are also in float16
model = model.to(torch.float16)

# Load dataset
dataset = load_dataset('hotpot_qa', 'distractor')

# Preprocessing function
def preprocess_data(input_data):
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

# Access the 'train' and 'validation' splits
train_dataset = dataset['train'].select(range(2000))
validation_dataset = dataset['validation'].select(range(2000))

# Apply preprocessing to the dataset
train_dataset = train_dataset.map(preprocess_data)
validation_dataset = validation_dataset.map(preprocess_data)

def tokenize_data(example):
    input_text = example['input_text']
    output_text = example['output_text']
    
    # Tokenize input and output texts
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
    outputs = tokenizer(output_text, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
    
    inputs['labels'] = outputs['input_ids']
    
    return inputs

# Apply tokenization to the dataset
train_dataset = train_dataset.map(tokenize_data, batched=True, remove_columns=["input_text", "output_text"])
validation_dataset = validation_dataset.map(tokenize_data, batched=True, remove_columns=["input_text", "output_text"])

# Disable cache for training
model.config.use_cache = False

# Training arguments
training_args = TrainingArguments(
    output_dir='./output',
    per_device_train_batch_size=4,  # Increase batch size if GPU has memory capacity
    gradient_accumulation_steps=16,  # Reduce gradient accumulation for faster steps
    max_steps=2000,
    learning_rate=0.0002,
    weight_decay=0.0,
    max_grad_norm=0.3,
    gradient_checkpointing=False,  # Disable gradient checkpointing to speed up
    lr_scheduler_type='constant',
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy='steps',
    save_steps=250,
    report_to="tensorboard",
    logging_dir='./output/logs',
    fp16=True,  # Enable mixed precision training
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  
    eval_dataset=validation_dataset, 
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the model and tokenizer to Hugging Face hub
from huggingface_hub import create_repo, Repository

repo_name = "4-Bit-QLORA-Implementation"  # Choose a unique name
create_repo(repo_name, private=False)

model.save_pretrained('./4-Bit-QLORA-Implementation')
tokenizer.save_pretrained('./4-Bit-QLORA-Implementation')

# Clone the repository (if it's the first time, otherwise it's already cloned)
repo = Repository(local_dir='./4-Bit-QLORA-Implementation', clone_from=repo_name)

# Push the changes to the repository
with repo.commit("Add model"):
    trainer.save_model()
    tokenizer.save_pretrained(repo.local_dir)

repo.push_to_hub()
