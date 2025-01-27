import re
import os
import transformers
from tqdm import tqdm
from rouge_score import rouge_scorer
import numpy as np
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)

# Prompt templates for Alpaca (same used in the training)
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:"
        "\n{input}\n\n### Response:"),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"),
}

# Constants
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
ANSWER_TRIGGER = "The answer is"

# Extract answer from model output
def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip().replace(",", "")
        return match_str
    else:
        return INVALID_ANS

# Check if model answer is correct
def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    return model_answer == gt_answer

# Build prompt for model input
def build_prompt(instruction, input, prompt_input=PROMPT_DICT["prompt_input"], prompt_no_input=PROMPT_DICT["prompt_no_input"]):
    if input != "":
        input_text_prompt = prompt_input.format(instruction=instruction, input=input)
    else:
        input_text_prompt = prompt_no_input.format(instruction=instruction)
    return input_text_prompt

# Clean model prediction
def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    pred = preds[1] if answer_flag else preds[-1]
    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    if not pred:
        return INVALID_ANS
    pred = pred[0] if answer_flag else pred[-1]
    if pred[-1] == ".":
        pred = pred[:-1]
    return pred

# Evaluate ROUGE scores
def evaluate_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    print("Scores per sample:")
    for rouge_type, score in scores.items():
        print(f"{rouge_type}: Precision: {score.precision:.3f}, Recall: {score.recall:.3f}, F1 Score: {score.fmeasure:.3f}")
    return scores

# Main function adapted for alpaca_testset.jsonl
def main():
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)
    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # Initialize chatbot with the fine-tuned adapter
    fschatbot = FSChatBot(init_cfg)

    # Load test set
    test_file = f'{init_cfg.data.root}/alpaca_testset.jsonl' # It can be modified to use alpaca_valset.jsonl instead
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found.")
        return
    list_data_dict = load_jsonl(test_file, instruction='instruction', input='input', output='output')

    rouge1_f1_scores = []
    rouge2_f1_scores = []
    rougeL_f1_scores = []

    # Evaluate each sample
    for idx, sample in enumerate(tqdm(list_data_dict)):
        input_text = build_prompt(sample['instruction'], sample['input'])
        generate_kwargs = dict(max_new_tokens=init_cfg.llm.chat.max_len) 
        model_completion = fschatbot.generate(input_text, generate_kwargs)

        print(input_text)   
        print(model_completion + "\n")  
        print(f'### Reference:\n{sample["output"]}\n')
 
        scores = evaluate_rouge(sample["output"], model_completion)
        rouge1_f1_scores.append(scores['rouge1'].fmeasure)
        rouge2_f1_scores.append(scores['rouge2'].fmeasure)
        rougeL_f1_scores.append(scores['rougeL'].fmeasure)

        print(f'Num of total question: {idx+1}/1000\n')
        print("--------------------------------------------------------")

    # Print final scores
    print("Final ROUGE-1 score:", np.mean(rouge1_f1_scores))
    print("Final ROUGE-2 score:", np.mean(rouge2_f1_scores))
    print("Final ROUGE-L score:", np.mean(rougeL_f1_scores))
        
if __name__ == "__main__":
    main()
