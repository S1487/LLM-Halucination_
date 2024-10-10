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
   - Specify the evaluated task (qa, dialogue or summarization)
2. Choose one of the models to run evaluation on:
<img width="584" alt="Screenshot 2024-10-10 at 10 37 41 PM" src="https://github.com/user-attachments/assets/1d5a667e-4306-454b-b450-12f0c5079caa">
 ```
   cd LLM-Halucination_
   python evaluate.py –-task qa --model Llama2
```



 
