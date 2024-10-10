# LLM_Halucination

Steps for Reproduction
Cloud Services (Google Cloud): 
Create a Google Cloud Compute Instance. This requires activation of your Google Cloud account from free-tier to non-free tier. These are the recommended specifications for the instance:
Region: asia-east-1 (a or c)
GPU: NVIDIA L4 (Initially, a GPU quota increase may need to be requested, which is usually approved within a few minutes)
Machine Type: g2-standard-4 (2 core, 16GB memory)
Boot Disk: Deep Learning on Linux OS, version - Deep Learning VM with CUDA 11.8 M125. Size- 150-200GB. Balanced persistent boot disk type.
	(A100 GPU is more ideal, however this GPU is more difficult to acquire and is not available in most regions)

The model can then be started and accessed through SSH.

Model from github:
Clone our GitHub repository using “git clone https://github.com/S1487/LLM-Halucination_.git”.
Run the “requirements.txt” file to install necessary dependencies.
Using the meta-llama2 model requires access through HuggingFace. Request access using https://huggingface.co/meta-llama/Llama-2-7b-chat-hf, and create an access token that can be used to login from the Google Cloud Instance. 
Use “huggingface-cli login”, enter your huggingface access token created earlier to gain access to the models from the virtual machine. This only needs to be done once.
Evaluation code: 
Move into the evaluation directory and run the evaluation code:
“cd LLM-Halucination_”
“python evaluate.py –task qa” (Specify qa, dialogue or summarization)
 
Analysis code
There are some additional steps to ensure that the analysis code executes without error:
Reinstall specific numpy version:
“Pip uninstall numpy”
“Pip install numpy==1.26.4”
Download two modules:
“python -m nltk.downloader stopwords”
“python -m spacy download en_core_web_sm”

The analysis can now be executed:
Navigate to the analysis directory:
“cd analysis”
Run the analysis code:
python analyze.py --task qa --result ../evaluation/qa/qa_Llama-2-7b-chat-hf_results.json --category all
