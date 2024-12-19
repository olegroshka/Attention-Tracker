# Attention Tracker: Detecting Prompt Injection Attacks in LLMs

Welcome to the official repository for **"Attention Tracker: Detecting Prompt Injection Attacks in LLMs"**. This repository provides scripts and tools to identify important attention heads and evaluate prompt injection attacks on large language models (LLMs).

Project page: https://huggingface.co/spaces/TrustSafeAI/Attention-Tracker 

Paper: https://arxiv.org/abs/2411.00348 

---

## Features
- **Identify Important Heads**: Determine which attention heads are critical for detecting prompt injection attacks.
- **Run Experiments**: Execute experiments on datasets to evaluate the model's effectiveness.
- **Test Queries**: Test individual queries against your chosen model.

---

## Getting Started

### Prerequisites
1. Ensure you have Python installed.
2. Install the necessary dependencies listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Find Important Heads
    To identify important attention heads, run:
    ```bash
    ./scripts/find_heads.sh
    ```
2. To specify important heads, go to `configs/model_configs/{model_name}.json` and change the `["params"]["important_heads"]`
3. To run Experiments on [DeepSet Prompt Injection Dataset](https://huggingface.co/datasets/deepset/prompt-injections?row=19).
    ```bash
    ./scripts/run_dataset.sh
    ```
4. Test Individual Queries
    ```bash
    python run.py --model_name {model} --test_query "{query you want to test}"
    ```