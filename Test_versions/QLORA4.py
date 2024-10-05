import datasets
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
import torch
from accelerate import Accelerator
import bitsandbytes as bnb
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers.optimization import AdamW

# Initialize accelerator
accelerator = Accelerator()

# Class to hold arguments
class Args:
    bits = 4 
    float16 = True  

args = Args()

# 4-bit quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,   # Enable double quantization for efficiency
    bnb_4bit_compute_dtype=torch.float16,  # Make sure compute dtype matches your hardware precision
)

# Load model and tokenizer from Hugging Face
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    trust_remote_code=False,
    torch_dtype=torch.float16,  # Use float16 precision throughout
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
optimizer = AdamW(model.parameters(), lr=0.0001)

# Prepare model and optimizer with accelerator (no manual .to(device))
model, optimizer = accelerator.prepare(model, optimizer)

# LoRA configuration for QLoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Modify this based on the architecture you're fine-tuning
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA configuration to the model
model = get_peft_model(model, lora_config)
for name, param in model.named_parameters():
    if 'lora' not in name:  # Freeze base model parameters
        param.requires_grad = False

# Force the LoRA layers to use float16 precision if necessary
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        module.to(torch.float16)  # Set to float16 precision if required

# Load dataset
dataset = load_dataset('hotpot_qa', 'distractor')

# Preprocessing function for dataset
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
train_dataset = dataset['train'].select(range(1000))
validation_dataset = dataset['validation'].select(range(1000))

# Apply preprocessing to the dataset
train_dataset = train_dataset.map(preprocess_data)
validation_dataset = validation_dataset.map(preprocess_data)

# Tokenize dataset
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
import torch
print(torch.cuda.is_available())
model = model.to('cuda')
# Training arguments
training_args = TrainingArguments(
    output_dir='./output',
    per_device_train_batch_size=4,  # Increase batch size based on memory capacity
    gradient_accumulation_steps=16,
    max_steps=2000,
    learning_rate=0.0002,
    weight_decay=0.0,
    max_grad_norm=0.3,
    gradient_checkpointing=False,  # Disable checkpointing to speed up
    lr_scheduler_type='constant',
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy='steps',
    save_steps=250,
    report_to="tensorboard",  # Enable Tensorboard
    logging_dir='./output/logs',
)
from transformers import Trainer

class CustomTrainer(Trainer):
    def training_step(self, model, inputs):
        # Call the original training step method
        loss = super().training_step(model, inputs)
        
        # Print or log info here (batch by batch)
        self.state.global_step += 1  # Keep track of global steps
        print(f"Completed step {self.state.global_step} with loss {loss.item()}")
        
        return loss
# Initialize the trainer
trainer = CustomTrainer(
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

repo_name = "4-Bit-QLORA-Implementation"
create_repo(repo_name, private=False)

model.save_pretrained('./4-Bit-QLORA-Implementation')
tokenizer.save_pretrained('./4-Bit-QLORA-Implementation')

repo = Repository(local_dir='./4-Bit-QLORA-Implementation', clone_from=repo_name)

# Push the model to the hub
with repo.commit("Add model"):
    trainer.save_model()
    tokenizer.save_pretrained(repo.local_dir)

repo.push_to_hub()
