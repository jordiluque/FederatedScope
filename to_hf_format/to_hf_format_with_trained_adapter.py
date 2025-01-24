from safetensors.torch import save_model
from huggingface_hub import login
import torch
import argparse

from federatedscope.llm.model.model_builder import get_model_from_huggingface
from federatedscope.llm.dataloader import get_tokenizer
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.core.configs.config import global_cfg

def create_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("-cfg", "--cfg_file", type=str,
                            help="Config file in yaml format")
        parser.add_argument("-hf", "--hf_token", type=str,
                            help="Huggingface token")
        parser.add_argument("-o", "--out_dir", type=str,
                            help="Output directory")
        return parser

parser = create_parser()
args = parser.parse_args()

# Initialize configuration
init_cfg = global_cfg.clone()
init_cfg.merge_from_file(args.cfg_file)

model_name = init_cfg.model.type.split("@")[0]
root = init_cfg.data.root
tok_len = init_cfg.llm.tok_len
model_hub = init_cfg.model.type.split("@")[1]
path_to_trained_adapt_ckpt = init_cfg.federate.save_to

# Login to Hugging Face
login(args.hf_token)

# Get tokenizer and base model
tokenizer , _ = get_tokenizer(model_name, root, tok_len, model_hub)
base_model = get_model_from_huggingface(model_name, init_cfg)

# Get base model with adapter
base_model_with_adapt = get_llm(init_cfg)

# Load trained adapter checkpoint
ckpt = torch.load(path_to_trained_adapt_ckpt, map_location='cpu', weights_only=True)
base_model_with_adapt.load_state_dict(ckpt['model'], strict=False)

# Save base model + trained adapter
out_dir = f"{args.out_dir}/model.safetensors"
save_model(base_model_with_adapt, out_dir)
