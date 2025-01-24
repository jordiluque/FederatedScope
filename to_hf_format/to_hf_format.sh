#!/bin/bash

work_dir="_path_to_FederatedScope_dir"
hf_model_path="microsoft/Phi-3.5-mini-instruct"
model_name=$(basename "$hf_model_path")
org=$(dirname "$hf_model_path")
hf_token="your_hugging_face_token"
hash_code_model_snapshot="hash_code_of_your_model_snapshot"
clients=(1 3 6 10 15 20 30)


for c in "${clients[@]}"; do
    experiment_name="ds_${c}c_1000r_30ls"

    # Path to the yaml config file
    cfg_file="${work_dir}/configs/comparable/standalone/${model_name}/${experiment_name}.yaml"
    out_dir="to_hf_format/hf_format/comparable/standalone/${model_name}/${experiment_name}"
    mkdir -p $out_dir

    # Convert the model (with the trained adapter embeded) to .safetensors format (Hugging Face format)
    python to_hf_format/to_hf_format_with_trained_adapter.py \
        --cfg_file $cfg_file \
        --hf_token $hf_token \
        --out_dir $out_dir

    src_path_from_cache="$HOME/.cache/huggingface/hub/models--${org}--${model_name}/snapshots/${hash_code_model_snapshot}"
    dest_path_to_hf_format="${work_dir}/to_hf_format/hf_format/comparable/standalone/${model_name}/${experiment_name}"

    # Copy the files from the .cache to the destination (except the ones that finish with .safetensors)
    rsync -av --exclude='*.safetensors' $src_path_from_cache/* $dest_path_to_hf_format
done