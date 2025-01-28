#!/bin/bash

# Creates config YAML file for the FederatedScope experiments
mode="standalone"
client_num=6 # Number of clients for that specific experiment
total_round_num=1000
model_path="utter-project/EuroLLM-9B-Instruct"
model_name=$(basename "$model_path")
method="FedAvg" # FedAvg, global
dataset="alpaca"
eval_freq=10

# If we want to perform "comparable" experiments (i.e., same proportion of train/valid/test splits for each client)
comparable=true

if [ "$comparable" = true ]; then
	out_dir="comparable/${mode}/${model_name}"
  	# Set the maximum number of clients to perform "comparable" experiments 
	N_max_clients=30  
else
  out_dir="${mode}/${model_name}"
fi
mkdir -p $out_dir

# Desired train/val/test partition for each client
splits=(0.96 0.02 0.02)
train_split=${splits[0]}
val_split=${splits[1]}
test_split=${splits[2]}

# Function to calculate split values in FederatedScope config file to perform "comparable" experiments
calculate_new_splits() {
	local train_split=$1
	local val_split=$2
	local test_split=$3
    local N_max_clients=$4
    local n_clients=$5

	delta=$(echo "scale=10; $n_clients / $N_max_clients" | bc -l)
	ne_train_split=$(echo "scale=10; $train_split * $delta" | bc -l)
	new_val_split=$(echo "scale=10; $val_split * $delta" | bc -l)
	new_test_split=$(echo "scale=10; $test_split * $delta" | bc -l)
	echo "$new_train_split $new_val_split $new_test_split"
}

if [ "$comparable" = true ]; then
	read train_split val_split test_split < <(calculate_new_splits "$train_split" "$val_split" "$test_split" "$N_max_clients" "$client_num")
fi

# Create config yaml file
python create_config_yaml.py --mode ${mode} \
                --client_num $client_num \
                --total_round_num $total_round_num \
				--dataset "${dataset}" \
            	--model_path "${model_path}" \
				--method "${method}" \
				--eval_freq $eval_freq \
				--train_split $train_split \
				--val_split $val_split \
				--test_split $test_split \
				--out_dir $out_dir
echo "Config yaml created"