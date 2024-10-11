# Fine-Tuning Language Models to Reduce Model Hallucinations

This is the repo for our final year research project focusing on evaluating hallucinations in Large Language Models and analyzing the impact of fine-tuning for both in-domain and out-of-domain applications. This repository contains instructions and code for fine-tuning, evaluation and analysis of LLMs (LLama-2-7B, ChatGLM-6B, Falcon-7B).


Steps for Reproduction

## Environment Setup

**Google Services (Google Cloud)**

1. Create a Google Cloud Compute Instance. This requires activation of your Google Cloud account from free-tier to non-free tier. These are the recommended specifications for the instance:
   - Region: asia-east-1 (a or c)
   - GPU: NVIDIA L4 (Initially, a GPU quota increase may need to be requested, which is usually approved within a few minutes)
   - Machine Type: g2-standard-4 (2 core, 16GB memory)
   - Boot Disk: Deep Learning on Linux OS, version - Deep Learning VM with CUDA 11.8 M125. Size- 150-200GB. Balanced persistent boot disk type.

(A100 GPU is more ideal, however this GPU is more difficult to acquire and is not available in most regions) 2. The model can then be started and accessed through SSH.

## To run the model

**Model from github:**

1. Clone our GitHub repository using

```
git clone https://github.com/S1487/LLM-Halucination_.git
```

3. Run the “requirements.txt” file to install necessary dependencies.
4. Using the meta-llama2 model requires access through HuggingFace. Request access using https://huggingface.co/meta-llama/Llama-2-7b-chat-hf, and create an access token that can be used to login from the Google Cloud Instance.
5. Use “huggingface-cli login”, and enter your huggingface access token created earlier to gain access to the models from the virtual machine. This only needs to be done once.

**Evaluation code:**

1. Move into the evaluation directory and run the evaluation code:
   - Specify the evaluated task (qa, dialogue or summarization)
2. Choose one of the models to run evaluation on:
   <img width="584" alt="Screenshot 2024-10-10 at 10 37 41 PM" src="https://github.com/user-attachments/assets/1d5a667e-4306-454b-b450-12f0c5079caa">

```
   cd LLM-Halucination_
   python evaluate.py –-task qa --model Llama2
```

**Analysis Code**

There are some additional steps to ensure that the analysis code executes without error:

1. Reinstall specific numpy version:

```
pip uninstall numpy
pip install numpy==1.26.4
```

2. Download two modules:

```
python -m nltk.downloader stopwords
python -m spacy download en_core_web_sm
```

Now the analysis can now be executed:

1. Navigate to the analysis directory:

```
cd analysis
```

2. Run the analysis code:

```
python analyze.py --task qa --result ../evaluation/qa/qa_Llama-2-7b-chat-hf_results.json --category all
```

You can choose which task to run analysis on, and specify the path to the specific evaluation result files.
The complete setup and execution can be run on a local machine, given that the appropriate hardware resources are available (At least 16GB of GPU RAM.)
