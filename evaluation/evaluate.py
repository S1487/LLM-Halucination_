import random
import time
import json
import argparse
from transformers import AutoModelForCausalLM,AutoModel, AutoTokenizer, pipeline,BitsAndBytesConfig
import torch
from peft import PeftModel
from peft import get_peft_model
from peft import PeftConfig, get_peft_model

if torch.cuda.is_available():
    print("GPU is available. Device:", torch.cuda.get_device_name(0))
else:
    print("GPU is not available, using CPU.")

mod_req = input("Model: ")
if  mod_req =="Llama2":
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map="cuda",
                                                torch_dtype=torch.float16,
                                                trust_remote_code=False,
                                                revision="main")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
elif mod_req =="Q4-Llama2":
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0)
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map="cuda",
                                                quantization_config=quantization_config,
                                                trust_remote_code=False,
                                                torch_dtype = torch.bfloat16,
                                                revision="main")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
elif mod_req =="Q8-Llama2":
    quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0)
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map="cuda",
                                                quantization_config=quantization_config,
                                                trust_remote_code=False,
                                                torch_dtype = torch.bfloat16,
                                                revision="main")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
elif mod_req =="Q8-Llama2-QLORA":
    quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0)
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map="cuda",
                                                quantization_config=quantization_config,
                                                trust_remote_code=False,
                                                torch_dtype = torch.bfloat16,
                                                revision="main")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    adapter_name = "Parfaitx74/TestQLORA-Adapterv2"  # Replace with your adapter's path
    peft_config = PeftConfig.from_pretrained(adapter_name)


    model = get_peft_model(model, peft_config)


    model.eval()

elif mod_req =="Q4-Llama2-QLORA":
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0)
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map="cuda",
                                                quantization_config=quantization_config,
                                                trust_remote_code=False,
                                                torch_dtype = torch.bfloat16,
                                                revision="main")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the LoRA adapter from the Hugging Face Hub
    adapter_name = "Parfaitx74/TestQLORA-Adapterv2"  # Replace with your adapter's path
    # Load the LoRA adapter configuration
    peft_config = PeftConfig.from_pretrained(adapter_name)


elif mod_req =="Q4-Llama2-QLORA_low_dropout":
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0)
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map="cuda",
                                                quantization_config=quantization_config,
                                                trust_remote_code=False,
                                                torch_dtype = torch.bfloat16,
                                                revision="main")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the LoRA adapter from the Hugging Face Hub
    adapter_name = "Parfaitx74/TestQLORA-Adapter_Low_Dropout"  # Replace with your adapter's path
    # Load the LoRA adapter configuration
    peft_config = PeftConfig.from_pretrained(adapter_name)

elif  mod_req =="ChatGLM":
    model_name = "THUDM/chatglm-6b"
    model = AutoModel.from_pretrained(model_name,
                                      device_map="cuda",
                                      trust_remote_code=True).half().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

elif mod_req =="Falcon":
    model_name = "tiiuae/falcon-7b"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                              device_map="cuda",
                                              torch_dtype=torch.bfloat16,
                                              trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    
    
if mod_req == "ChatGLM":
    pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
else:
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    
    
def get_qa_response(model, question, answer, instruction):

    prompt = instruction + "\n\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:"
    while True:
        try:
            res = pipe(prompt,max_new_tokens=2)
            response = res[0]['generated_text'].strip()
            response = response.split("#Your Judgement#:")[-1].strip()
            break
        except Exception as e:
            print('Retrying...', str(e))
            time.sleep(20)
    
    return response


def get_dialogue_response(model, dialog, response, instruction):

    prompt = instruction + "\n\n#Dialogue History#: " + dialog + "\n#Response#: " + response + "\n#Your Judgement#:"
    while True:
        try:
            res = pipe(prompt,max_new_tokens=2)
            response = res[0]['generated_text'].strip()
            response = response.split("#Your Judgement#:")[-1].strip()
            break
        except Exception as e:
            print('Retrying...', str(e))
            time.sleep(20)


    return response


def num_tokens_from_message(message, model):
    encoding = AutoTokenizer.from_pretrained(model)
    num_tokens = len(encoding.encode(message))
    return num_tokens


def truncate_message(prompt1, prompt2, model):
    if num_tokens_from_message(prompt1 + prompt2, model) > 2033:
        truncation_length = 2033 - num_tokens_from_message(prompt2)
        while num_tokens_from_message(prompt1) > truncation_length:
            prompt1 = " ".join(prompt1.split()[:-1])
    prompt = prompt1 + prompt2
    return prompt


def get_summarization_response(model, document, summary, instruction):

    prompt1 = instruction + "\n\n#Document#: " + document
    prompt2 = "\n#Summary#: " + summary + "\n#Your Judgement#:"
    if model == "Llama-2-7b-chat":
        prompt = truncate_message(prompt1, prompt2)
    else:
        prompt = prompt1 + prompt2
    while True:
        try:
            res = pipe(prompt,max_new_tokens=2)
            response = res[0]['generated_text'].strip()
            response = response.split("#Your Judgement#:")[-1].strip()
            break
        except Exception as e:
            print('Retrying...', str(e))
            time.sleep(20)
    return response


def evaluation_qa_dataset(model, file, instruction, output_path):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        sample_limit = 10000
        for i in range(len(data)):
            if i >=sample_limit:
                break

            knowledge = data[i]["knowledge"]
            question = data[i]["question"]
            hallucinated_answer = data[i]["hallucinated_answer"]
            right_answer = data[i]["right_answer"]

            if random.random() > 0.5:
                answer = hallucinated_answer
                ground_truth = "Yes"
            else:
                answer = right_answer
                ground_truth = "No"

            ans = get_qa_response(model, question, answer, instruction)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
                incorrect += 1

            assert(gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            print('sample {} success......'.format(i))
            dump_jsonl(gen, output_path, append=True)

        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct/sample_limit))


def evaluation_dialogue_dataset(model, file, instruction, output_path):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        sample_limit = 10000
        for i in range(len(data)):
            if i >=sample_limit:
                break
            knowledge = data[i]["knowledge"]
            dialog = data[i]["dialogue_history"]
            hallucinated_response = data[i]["hallucinated_response"]
            right_response = data[i]["right_response"]

            if random.random() > 0.5:
                response = hallucinated_response
                ground_truth = "Yes"
            else:
                response = right_response
                ground_truth = "No"

            ans = get_dialogue_response(model, dialog, response, instruction)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
            assert (gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            print('sample {} success......'.format(i))
            dump_jsonl(gen, output_path, append=True)

        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct / sample_limit))


def evaluation_summarization_dataset(model, file, instruction, output_path):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        sample_limit = 10000
        for i in range(len(data)):
            if i >=sample_limit:
                break
            document = data[i]["document"]
            hallucinated_summary = data[i]["hallucinated_summary"]
            right_summary = data[i]["right_summary"]

            if random.random() > 0.5:
                summary = hallucinated_summary
                ground_truth = "Yes"
            else:
                summary = right_summary
                ground_truth = "No"

            ans = get_summarization_response(model, document, summary, instruction)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
            assert (gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            print('sample {} success......'.format(i))
            dump_jsonl(gen, output_path, append=True)

        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct / sample_limit))


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hallucination Generation")

    parser.add_argument("--task", default="qa", help="qa, dialogue, or summarization")
    parser.add_argument("--model", default="Llama-2-7b-chat-hf", help="model name")
    args = parser.parse_args()

    instruction_file = "{}/{}_evaluation_instruction.txt".format(args.task, args.task)
    f = open(instruction_file, 'r', encoding="utf-8")
    instruction = f.read()

    model = args.model
    output_path = "{}/{}_{}_{}results.json".format(args.task, args.task, args.model,mod_req)

    data = "../data/{}_data.json".format(args.task)

    if args.task == "qa":
        evaluation_qa_dataset(model, data, instruction, output_path)
    elif args.task == "dialogue":
        evaluation_dialogue_dataset(model, data, instruction, output_path)
    elif args.task == "summarization":
        evaluation_summarization_dataset(model, data, instruction, output_path)
    else:
        raise ValueError("The task must be qa, dialogue, or summarization!")
