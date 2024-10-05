from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
import transformers
import torch

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

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="cuda",
                                             torch_dtype=torch.float16,
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


trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=20,
        output_dir="finetuneresult/results",
        report_to="tensorboard",
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
)


trainer.train()
model.save_pretrained("finetuneresult/fine_tuned_llama2_7b_chat")
