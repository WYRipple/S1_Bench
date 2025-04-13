
import requests
import json
import pdb
import os
import re
import time
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from openai import OpenAI
from datetime import datetime
current_file_path = os.path.abspath(__file__)
base_directory = os.path.dirname(current_file_path)

def get_evaluator_response(prompt):    
    temperature = 0.3
    top_p = 0.9

    maxtry = 10     
    delay = 5      
        
    openai_api_key = "sk-xxx"
    openai_api_base = ""

    conversation = [{"role":"user", "content": prompt}]
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        
    i = 0
    while i < maxtry:
        try:
            chat_response = client.chat.completions.create(
                model="gpt-4o",
                messages=conversation,
                temperature=temperature,
                top_p=top_p
            )
            return chat_response.choices[0].message.model_dump()['content']
        
        except Exception as e:
            i += 1
            print(f"Error!\n{e}\n{i + 1} time, retry after {delay} seconds...")
            time.sleep(delay)
            continue
    return "Err..."


def read_jsonl_to_list(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                result.append(data)
    
    return result


def read_json_to_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        result = json.load(json_file)
    return result


def append_to_jsonl(file_path, data_dict):
    json_str = json.dumps(data_dict, ensure_ascii=False) + '\n'
    
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json_str)


def extract_json_from_llm_response(response):
    try:
        json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        match = re.search(json_pattern, response)
        
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = response.strip()
        
        json_dict = json.loads(json_str)
        return json_dict
    
    except json.JSONDecodeError as e:
        print(f"JSON decode err: {e}")
        return None
    except Exception as e:
        print(f"JSON exception err: {e}")
        return None



def main(_input):
    sample, save_file = _input
    
    eval_prompt_path = base_directory+"/eval_acc_prompt.txt"
    with open(eval_prompt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    change_prompt_path = base_directory+"/rewrite_json.txt"
    with open(change_prompt_path, 'r', encoding='utf-8') as file:
        content_change = file.read()
    

    max_attempts = 30
    attempt = 0
    success = False
    if sample["think_success"] not in [300, 301]:
        if sample["model_response"] != "":
            eval_target = sample["answering_part"]
        else:
            eval_target = sample["thinking_part"]

        while attempt < max_attempts and not success:
            attempt += 1
            prompt = content.format(question=sample["question"], model_answer=eval_target, ground_truth=sample["answer"])

            eval_response = get_evaluator_response(prompt)
            eval_response_json = extract_json_from_llm_response(eval_response)

            if eval_response_json is not None and isinstance(eval_response_json, dict) and 'explain' in eval_response_json and 'result' in eval_response_json:
                sample['eval_exp'] = eval_response_json['explain']
                sample['eval_result'] = eval_response_json['result']
                append_to_jsonl(save_file, sample)
                success = True
            else:
                # 调用修改prompt
                prompt_change = content_change.format(llm_response=eval_response)
                eval_response_rewrite = get_evaluator_response(prompt_change)
                eval_response_json = extract_json_from_llm_response(eval_response_rewrite)
                if eval_response_json is not None and isinstance(eval_response_json, dict) and 'explain' in eval_response_json and 'result' in eval_response_json:
                    sample['eval_exp'] = eval_response_json['explain']
                    sample['eval_result'] = eval_response_json['result']
                    append_to_jsonl(save_file, sample)
                    success = True
                else:
                    # JSON extraction failed, retry
                    print(f"json decode error, try again")
    else:
        success = True
        sample['eval_exp'] = ""
        sample['eval_result'] = -1
        append_to_jsonl(save_file, sample)


    if not success:
        # All attempts failed, save sample with default values
        sample['eval_exp'] = ""
        sample['eval_result'] = -1  # Default to incorrect answer
        append_to_jsonl(save_file, sample)


def init_main(file_path, save_path):
    use_async = True        
    pro_num = 10            

    data_list = read_json_to_list(file_path)
    print(f"load eval data num = {len(data_list)}")

    input_data_list = []
    for sample in data_list:
        input_data_list.append([sample, save_path])

    if use_async:
        func = partial(main)
        with Pool(processes=pro_num) as pool:
            for _ in tqdm(pool.imap(func, input_data_list), total=len(input_data_list)):
                pass
    else:
        for this_sample in tqdm(input_data_list):
            main(this_sample)


# 主函数
if __name__ == "__main__":
    model_list = ["DS-R1-1.5B", "DS-R1-7B", "DS-R1-8B", "DS-R1-14B", "DS-R1-32B", "DS-R1-70B", "QwQ-32B", "L-R1-7B-DS", 
                    "L-R1-14B-DS", "L-R1-32B-DS", "L-R1-32B", "s1.1-7B", "s1.1-14B", "s1.1-32B", 
                    "EXAONE-2.4B", "EXAONE-7.8B", "EXAONE-32B", "Nemotron-8B", "Nemotron-49B", "Sky-T1-32B", "DS-R1", "Hunyuan-T1"]
    
    for model_name in model_list:
        print(f"top-p eval: [ {model_name} ]")
        for i in range(5):
            file_path = base_directory + f"/LRM_split/{model_name}/LRM_response_split_{model_name}_{str(i)}.json"
            save_path = base_directory + f"/LRM_acc_eval/{model_name}/LRM_response_eval_{model_name}_{str(i)}.json"
            print(f"Starting new evaluation. Results will be saved in {save_path}")
            init_main(file_path, save_path)
    
    for model_name in model_list:
        print(f"greedy eval: [ {model_name} ]")
        for i in range(1):
            file_path = base_directory + f"/LRM_split_temp_0/{model_name}/LRM_response_split_{model_name}_{str(i)}.json"
            save_path = base_directory + f"/LRM_acc_eval_temp_0/{model_name}/LRM_response_eval_{model_name}_{str(i)}.json"
            print(f"Starting new evaluation. Results will be saved in {save_path}")
            init_main(file_path, save_path)

