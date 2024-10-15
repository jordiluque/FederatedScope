# Installation, Setup and Running of FederatedScope for LLMs Fine-tuning

First, use a virtual environment manager such as pyenv to create a virtual environment. Make sure you are using Python 3.9.0:

```bash
pyenv install 3.9.0
pyenv virtualenv 3.9.0 fs-llm_3.9.0
pyenv activate fs-llm_3.9.0
```

Clone the specific branch of the FederatedScope repository to your machine:

```bash
git clone --branch llm-eloquence https://github.com/jordiluque/FederatedScope.git
```

To ensure that the correct CUDA paths are set, add the following lines to your `.bashrc` (or equivalent shell configuration file). The CUDA version should be around version 12 (e.g., 12.4, 12.5, or 12.6). If you don’t already have the [CUTLASS](https://github.com/NVIDIA/cutlass) repository installed, clone and set it up on your machine.

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

Install the following Python libraries. The specific versions are known to work well:
```bash
pip install torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1
```

From the source of the repository, install the required FederatedScope requirements:
```bash
pip install -e .[llm]
```

Check if the default script runs correctly to verify the installation:

```bash
python federatedscope/main.py --cfg federatedscope/llm/baseline/testcase.yaml
```

Now let's install and configure DeepSpeed. It is is highly recommended for efficiently fine-tuning LLMs. To install it, run:

```bash
pip install deepspeed
```

Install the cupy library for CUDA Acceleration with CUDA 12 support:

```bash
pip install cupy-cuda12x
```

If you are working with recent models (e.g., Phi models), they may not be included in the default version of the transformers library. In this case, upgrade the library:

```bash
pip install --upgrade transformers
```

Before using DeepSpeed, review the configuration file at `federatedscope/llm/baseline/deepspeed/ds_config_4bs.json`. Ensure that the train_batch_size parameter is properly set to match the number of GPUs available on your machine.

Check if fine-tuning an LLM in standalone mode works correctly with DeepSpeed. Run the following script to verify that the fine-tuning process is functioning properly:

```bash
deepspeed federatedscope/main.py --cfg configs/standalone/Phi-3.5-mini-instruct/ds_3c_200r_30ls.yaml
```

To execute federated fine-tuning in distributed mode, separate commands need to be run for the server and each client. In the FederatedScope framework, each client must run on a different machine. The following config files will allow us to test if the setup works with two clients in distributed mode. However, before running the commands, ensure that the `server_host`, `server_port`, `client_host`, and `client_port` fields in the config files are updated with the correct IP addresses and ports for your machines. Additionally, adjust CUDA_VISIBLE_DEVICES to reflect the number of GPUs available on each machine.

To run the server use:
```bash
deepspeed --master_addr=127.0.0.1 --master_port=29500 federatedscope/main.py --cfg configs/distributed/Phi-3.5-mini-instruct/server_ds_2c_200r_30ls.yaml
```

To run a first client in one machine use:
```bash
CUDA_VISIBLE_DEVICES=0,1,2 deepspeed --master_addr=127.0.0.1 --master_port=29500 federatedscope/main.py --cfg configs/distributed/Phi-3.5-mini-instruct/client_1_ds_2c_200r_30ls.yaml
```

To run a second client in another machine:
```bash
CUDA_VISIBLE_DEVICES=0,1,2 deepspeed --master_addr=127.0.0.1 --master_port=29500 federatedscope/main.py --cfg configs/distributed/Phi-3.5-mini-instruct/client_1_ds_2c_200r_30ls.yaml 
```