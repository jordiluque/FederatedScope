# How to install FederatedScope and make it run?

## Installation and setup 

### 1. Set Up a Virtual Environment

First, use a virtual environment manager such as pyenv to create a virtual environment. Make sure you are using Python 3.9.0:

```bash
pyenv install 3.9.0
pyenv virtualenv 3.9.0 fs-llm_3.9.0
pyenv activate fs-llm_3.9.0
```

### 2. Clone the FederatedScope Repository

Clone the specific branch of the FederatedScope repository to your machine:

```bash
git clone --branch llm-eloquence https://github.com/jordiluque/FederatedScope.git
```

### 3. Configure CUDA Environment Variables

To ensure that the correct CUDA paths are set, add the following lines to your .bashrc (or equivalent shell configuration file). The CUDA version should be around version 12 (e.g., 12.4, 12.5, or 12.6). If you don’t already have the [CUTLASS](https://github.com/NVIDIA/cutlass) repository installed, clone and set it up on your machine.

```bash
export PATH=/usr/local/cuda-12/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:/usr/local/cuda-12/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12
export CUTLASS_PATH=/home/user/repos/cutlass 
```

After editing `.bashrc`, don't forget to run:

```bash
source ~/.bashrc
```

### 4. Install Python Dependencies

Install the following Python libraries. The specific versions are known to work well:
```bash
pip install torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1
```

### 5. Install FederatedScope Requirements

From the source of the repository, install the required dependencies:
```bash
pip install -e .[llm]
```

### 6. Verify Installation

Check if the default script runs correctly:

```bash
python federatedscope/main.py --cfg federatedscope/llm/baseline/testcase.yaml
```

### 7. Install and Configure DeepSpeed (Recommended for Fine-tuning LLMs)

DeepSpeed is highly recommended for efficiently fine-tuning LLMs. To install it, run:

```bash
pip install deepspeed
```

### 8. Install CuPy for CUDA Acceleration

Install the cupy library with CUDA 12 support:

```bash
pip install cupy-cuda12x
```

### 9. Update the Transformers Library (for Recent Models)

If you are working with recent models (e.g., Phi models), they may not be included in the default version of the transformers library. In this case, upgrade the library:

```bash
pip install --upgrade transformers
```

### 10. DeepSpeed Configurations

Before using DeepSpeed, review the configuration file at `federatedscope/llm/baseline/deepspeed/ds_config_4bs.json`. Ensure that the train_batch_size parameter is properly set to match the number of GPUs available on your machine.

11. Test Fine-tuning an LLM with DeepSpeed

Check if fine-tuning an LLM in standalone mode works correctly with DeepSpeed. Run the following script to verify that the fine-tuning process is functioning properly:

```bash
deepspeed federatedscope/main.py --cfg configs/standalone/Phi-3.5-mini-instruct/ds_3c_200r_30ls.yaml
```