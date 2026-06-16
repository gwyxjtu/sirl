import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Can revert to multi-GPU once model loads
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'  # Use mirror for better connectivity
os.environ['no_proxy'] = 'hf-mirror.com,huggingface.co' 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
warnings.filterwarnings("ignore")
import subprocess
import json
import time
import sys
from utils import load_jsonl, extract_code_block, extract_obj, change_variable_types
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate
from rule_prompt_utils import gurobi_prompt_temp

# load checkpoints and tokenizer
# model_path = 'chenyitian-shanshu/SIRL-Gurobi'
model_path = './SIRL-Gurobi'
tensor_parallel_size = 2
solver_name = 'gurobi'
print("Loading model", model_path)
model = LLM(
    model=model_path,
    tensor_parallel_size=tensor_parallel_size,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    trust_remote_code=True,
    # max_model_len=4096
)
print("Model initialized.")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# load prompt template and functions for generation
zeroshot_prompt_system = PromptTemplate.from_template(gurobi_prompt_temp['system'])
zeroshot_prompt_user = PromptTemplate.from_template(gurobi_prompt_temp['user'])

def mp_worker(item):
    prompt = [
        {
            "role": "system",
            "content": zeroshot_prompt_system.format(question=item['en_question']).strip()
        },
        {
            "role": "user",
            "content": zeroshot_prompt_user.format(question=item['en_question']).strip()
        }
    ]
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return text

def generate_with_model(model, prompt, sampling_params):
    response = model.generate(prompt, sampling_params)
    result_text = [g.outputs[0].text for g in response]
    return result_text

# Load decode strategy
topk = 1
max_tokens = 16384
repetition_penalty = 1.02 # To avoid the occasional occurrence of repeated tokens
stop_tokens = ["</s>"]

# top-p strategy
sampling_params = SamplingParams(
    n=topk,
    temperature=0.5,
    top_p=0.9,
    max_tokens=max_tokens,
    stop=stop_tokens,
    repetition_penalty=repetition_penalty
)

# check the pass@1 accuracy
def check_result(result_str, item, solver_name='gurobi'):
    sub_answer = item['en_answer']
    # Convert sub_answer to float or None
    sub_answer = None if sub_answer == "No Best Solution" or "-9999" in str(sub_answer) else float(sub_answer)

    # Extract code snippet
    code_snippet = extract_code_block(result_str, solver_name)
    if not code_snippet:
        return 2

    # Run code snippet
    try:
        result = subprocess.run([sys.executable, '-c', code_snippet], capture_output=True, text=True, encoding='utf-8', timeout=200)
    except subprocess.TimeoutExpired:
        return 1 if sub_answer is None else 0

    # Check if execution failed
    if result.returncode != 0:
        return 3

    # Extract solver result
    solver_result = extract_obj(result.stdout,solver_name)

    # check the first time
    if solver_result is not None and sub_answer is not None and np.abs(solver_result - sub_answer) / (np.abs(sub_answer) + 1) <= 1e-6:
        return 1
    # Handle infeasible case or numerical mismatch since we ignore the variable types error
    if 'nfeasible' in result.stdout or (solver_result is not None and sub_answer is not None and np.abs(solver_result - sub_answer) / (np.abs(sub_answer) + 1) > 1e-6):
        # Try re-running with modified variables: we ignore the variable types error
        result_str = change_variable_types(result_str) # change the type of variables
        if result_str:
            try:
                code_snippet = extract_code_block(result_str, solver_name)
                result = subprocess.run([sys.executable, '-c', code_snippet], capture_output=True, text=True, encoding='utf-8', timeout=200)
                if result.returncode == 0:
                    new_result = extract_obj(result.stdout,solver_name)
                    if 'nfeasible' not in result.stdout: # infeasible and Infeasible
                        if new_result is not None and sub_answer is not None and np.abs(new_result - sub_answer) / (np.abs(sub_answer) + 1) < 1e-6:
                            return 1
                        if new_result == sub_answer:
                            return 1
            except subprocess.TimeoutExpired:
                print("over_time")
                return 1 if sub_answer is None else 0

    # Handle infeasible cas 0e after retry
    if 'nfeasible' in result.stdout:
        return 1 if sub_answer is None else 0

    # Final comparison
    if solver_result is not None and sub_answer is not None:
        return 1 if np.abs(solver_result - sub_answer) / (np.abs(sub_answer) + 1) < 1e-6 else 0
    return 1 if solver_result == sub_answer else 0

if __name__ == "__main__":
    # Ensure log directory exists
    os.makedirs("log", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join("log", timestamp)
    os.makedirs(run_log_dir, exist_ok=True)
    
    main_log_file = os.path.join(run_log_dir, "execution.log")
    
    def log_and_print(msg):
        print(msg)
        with open(main_log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # if you want to check pass@1 accuracy, please run this cell
    # Test the checkpoint
    datapath = 'test_data'
    testdataset = ['NL4OPT.jsonl', 'MAMO_EasyLP.json', 'MAMO_ComplexLP_revised.json', 'IndustryOR_fixedV2.json', 'OptMATH_Bench_193.jsonl', 'OptMATH_Bench_166.jsonl','OptiBench.jsonl']
    testdataset = ['EnergyLLM_dataset.json']
    for filepath in testdataset:
        dataset_name = filepath.replace(".jsonl", "").replace(".json", "")
        dataset_log_dir = os.path.join(run_log_dir, dataset_name)
        os.makedirs(dataset_log_dir, exist_ok=True)

        # loading data
        log_and_print(f'Loading data {filepath}')
        full_path = os.path.join(datapath, filepath)
        # Try loading as standard JSON first, then JSONL
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        except json.JSONDecodeError:
            test_data = load_jsonl(full_path)
        
        if 'IndustryOR' in filepath:
            test_data = [item for item in test_data if item.get('difficulty') == 'Hard']
            log_and_print(f'Filtered to {len(test_data)} Hard problems')
            
        log_and_print('Finish Loading')

        # generation
        prompt_list = []
        for item in test_data:
            prompt_list.append(mp_worker(item))
        result_strs = generate_with_model(model, prompt_list, sampling_params)
        
        snippet_package_cor = []
        
        # Process each item and save to its own folder
        for idx, (result_str, item) in enumerate(zip(result_strs, test_data)):
            # Create problem-level directory: index_ID
            prob_id = item.get('id', idx)
            prob_dir = os.path.join(dataset_log_dir, f"{idx:03d}_{prob_id}")
            os.makedirs(prob_dir, exist_ok=True)
            
            # 1. Save raw model output
            with open(os.path.join(prob_dir, "model_output.txt"), "w", encoding="utf-8") as f:
                f.write(result_str)
            
            # 2. Save problem reference (question and answer)
            with open(os.path.join(prob_dir, "ref.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "question": item.get('en_question', ''),
                    "answer": item.get('en_answer', '')
                }, f, indent=4, ensure_ascii=False)
                
            # 3. Extract and save code snippet
            code_snippet = extract_code_block(result_str, solver_name)
            if code_snippet:
                with open(os.path.join(prob_dir, "solution.py"), "w", encoding="utf-8") as f:
                    f.write(code_snippet)
            
            # 4. Check result and record status
            status = check_result(result_str, item, solver_name)
            snippet_package_cor.append(status)
            
            status_map = {1: "PASS", 0: "FAIL", 2: "NO_CODE_FOUND", 3: "EXECUTION_ERROR"}
            with open(os.path.join(prob_dir, "result.txt"), "w", encoding="utf-8") as f:
                f.write(f"Status: {status_map.get(status, 'UNKNOWN')}\n")

        # Dataset summary
        result = np.bincount(snippet_package_cor)
        log_and_print(f'Numbers of test cases in dataset {filepath}: {sum(result)}')
        log_and_print(f'Numbers of pass@1 cases in dataset {filepath}: {result[1]}')
        log_and_print(f'pass@1 accuracy for dataset {filepath}: {result[1]}/{sum(result)} = {result[1] / sum(result)}')
        log_and_print('-------------------------------------------------------------------')

    # # if you want to check pass@8 accuracy, please run this cell
    # # Test the checkpoint
    # datapath = 'test_data'
    # testdataset = ['NL4OPT.jsonl', 'MAMO_EasyLP.json', 'MAMO_ComplexLP_revised.json', 'IndustryOR_fixed.json', 'OptMATH_Bench_193.jsonl', 'OptMATH_Bench_166.jsonl','OptiBench.jsonl']
    # for filepath in testdataset:

    #     # loading data
    #     print('Loading data', filepath)
    #     test_data = [i for i in load_jsonl(os.path.join(datapath, filepath)) for _ in range(8)]
    #     print('Finish Loading')

    #     # generation

    #     prompt_list = []
    #     for item in test_data:
    #         prompt_list.append(mp_worker(item))
    #     result_strs = generate_with_model(model, prompt_list, sampling_params)
    #     snippet_package_cor = []
    #     score = []
    #     snippet_package_tmp=[]
    #     # check the pass@8 accuracy

    #     result_chunks = [result_strs[i:i + 8] for i in range(0, len(result_strs), 8)]
    #     test_data_chunks = [test_data[i:i + 8] for i in range(0, len(test_data), 8)]
    #     for result_chunk, items in zip(result_chunks,test_data_chunks):
    #         for chunk, item in zip(result_chunk, items):
    #             snippet_package_tmp.append(check_result(chunk, item, solver_name))
    #         if 1 in snippet_package_tmp:
    #             snippet_package_cor.append(1)
    #         else:
    #             snippet_package_cor.append(0)
    #         snippet_package_tmp.clear()
    #     result = np.bincount(snippet_package_cor)
    #     print(f'Numbers of test cases in dataset {filepath}: {sum(result)}')
    #     print(f'Numbers of pass@8 cases in dataset {filepath}: {result[1]}')
    #     print(f'pass@8 accuracy for dataset {filepath}: {result[1]}/{sum(result)} = {result[1] / sum(result)}')
    #     print('-------------------------------------------------------------------')