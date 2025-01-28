import argparse
import yaml

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        help="Standalone or distributed mode")
    parser.add_argument("--client_num", type=int, required=True,
                        help="Number of clients")
    parser.add_argument("--total_round_num", type=int, required=True,
                        help="Total number of federated learning rounds")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset for federated training")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained model")
    parser.add_argument("--method", type=str, required=True,
                        help="Federate method type: FedAvg, FedOpt, global...")
    parser.add_argument("--train_split", type=float, required=True,
                        help="Train split")
    parser.add_argument("--val_split", type=float, required=True,
                        help="Validation split")
    parser.add_argument("--test_split", type=float, required=True,
                        help="Test split")
    parser.add_argument("--eval_freq", type=int, required=True,
                        help="Frequency of evaluation rounds")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory to save the yaml config file.")
    return parser

def update_yaml(template_stand_dir, out_dir, mode, client_num, total_round_num, dataset, method, train_split, val_split, test_split, model_path, eval_freq):
    # Open the YAML file
    with open(template_stand_dir, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # Update the YAML fields
    yaml_data['expname_tag'] = f"ds_{client_num}c_{total_round_num}r_30ls"
    yaml_data['federate']['method'] = f"{method}"
    yaml_data['federate']['mode'] = f"{mode}"
    yaml_data['federate']['client_num'] = client_num
    yaml_data['federate']['total_round_num'] = total_round_num
    yaml_data['federate']['save_to'] = f"models/comparable/{mode}/{model_path.split('/')[1]}/ds_{client_num}c_{total_round_num}r_30ls.ckpt"
    yaml_data['data']['type'] = f"{dataset}@llm"
    yaml_data['data']['splits'] = [train_split, val_split, test_split]
    yaml_data['model']['type'] = f"{model_path}@huggingface_llm"
    yaml_data['eval']['freq'] = eval_freq

    # Write the changes back to the file
    with open(out_dir, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    template_stand_dir = "template_standalone.yaml"
    model_name = args.model_path.split('/')[1]
    out_dir = f"{args.out_dir}/ds_{args.client_num}c_{args.total_round_num}r_30ls.yaml"

    update_yaml(template_stand_dir, out_dir, args.mode, args.client_num, args.total_round_num, args.dataset, args.method, args.train_split, args.val_split, args.test_split, args.model_path, args.eval_freq)
