# LLM_Halucination

 Steps for Reproduction

 # To run the model

__Model from github:__

1. Clone our GitHub repository using “git clone https://github.com/S1487/LLM-Halucination_.git”.
2. Run the “requirements.txt” file to install necessary dependencies.
3. Using the meta-llama2 model requires access through HuggingFace.Request access using https://huggingface.co/meta-llama/Llama-2-7b-chat-hf, and create an access token that can be used to login from the Google Cloud Instance.
4. Use “huggingface-cli login”, enter your huggingface access token created earlier to gain access to the models from the virtual machine. This only needs to be done once.

   
__Evaluation code:__

1. Move into the evaluation directory and run the evaluation code:
   - “cd LLM-Halucination_”
```
   python evaluate.py –task qa 
```
   - (Specify qa, dialogue or summarization)
2. Choose one of the models to run evaluation on:

Model
Command parameter
 - Llama-2-7b-chat: “Llama2”
 - Llama-2-7b-chat - Q4: “Q4-Llama2”
 - Llama-2-7b-chat - Q8: “Q8-Llama2”
 - Llama-2-7b-chat - Q4_QLORA: “Q4-Llama2-QLORA”
 - Llama-2-7b-chat - Q8_QLORA: “Q8-Llama2-QLORA”
 - Llama-2-7b-chat - Q4_QLORA_low_dropout: “Q4-Llama2-QLORA_low_dropout”
 - ChatGLM-6B: “ChatGLM”
 - Falcon-7B: “Falcon”

 
