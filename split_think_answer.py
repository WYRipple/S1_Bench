import requests
import json
import pdb
import time
import os
import ast
import re
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from openai import OpenAI
from datetime import datetime
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
base_directory = os.path.dirname(current_file_path)

tokenizer = AutoTokenizer.from_pretrained("LLMs_tokenizer/qwen2.5")

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


def count_tokens(text):
    tokens = tokenizer.tokenize(text)
    token_count = len(tokens)
    return token_count



def false_type(think_success):
    if think_success == -1:
        return "There is one </think>, no thinking process, but with an answer."
    elif think_success == -2:
        return "There is one </think>, with a thinking process, but without an answer."
    elif think_success == -3:
        return "There are multiple </think>, with a thinking process and an answer."
    elif think_success == -4:
        return "There are multiple </think>, with a thinking process, but without an answer."
    elif think_success == -5:
        return "There are multiple </think>, no thinking process, but with an answer."

    elif think_success == -6:
        return "There is one or more </think> or special markers, no thinking process, and no answer."

    elif think_success == -7:
        return "There is one or more other special markers, with a thinking process and an answer."
    elif think_success == -8:
        return "There is one or more other special markers, with a thinking process, but without an answer."
    elif think_success == -9:
        return "There is one or more other special markers, no thinking process, but with an answer."

    elif think_success == -10:
        return "There are no special markers, and the maximum length is not reached."
    elif think_success == -11:
        return "There are no special markers, but the maximum length is reached."
    
    else:
        return "Something Err"

def test_print(sample, think, answer, think_success, use_error_print):
    if use_error_print:
        num_token = count_tokens(sample["model_response"])
        print(44*"*" + "Response Type Start" + 44*"*")
        response = sample["model_response"]
        print(f"{response}")
        print(100*"=")
        print(f"Type: {false_type(think_success)}")
        print(100*"-")
        print(f"token num = {num_token}")
        print(44*"*" + "Response Type End" + 44*"*")
        pdb.set_trace()


def is_empty_after_cleaning(input_string):
    cleaned_string = re.sub(r'<[^>]*>', '', input_string)
    cleaned_string = cleaned_string.replace(" ", "").replace("\n", "")
    return len(cleaned_string) == 0


def judge_special_flag(sample, use_error_print):

    special_markers = [
        "**答案**",
        "**Final Answer**",
        "</th think>",
        "</ think>",
        "</ reason>",
        "\nanswer\n"
    ]


    if count_tokens(sample["model_response"]) < 9000:

        if sample["model_response"].count("</think>") == 1:

            index = sample["model_response"].find("</think>")
            think = sample["model_response"][:index].strip()

            index += len("</think>")
            answer = sample["model_response"][index:].strip()

            if len(think) == 0 and len(answer) != 0:
                think_success = -1
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) != 0 and len(answer) == 0:
                think_success = -2
                test_print(sample, think, answer, think_success, use_error_print)
            else:
                think_success = 1
        
        elif sample["model_response"].count("</think>") > 1:
            indices = []
            start = 0
            while True:
                index = sample["model_response"].find("</think>", start)
                if index == -1:
                    break
                indices.append(index)
                start = index + len("</think>")
            
            think_parts = []
            for i in range(len(indices) - 1):
                
                if i == 0:
                    think_parts.append(sample["model_response"][:indices[i]].strip())
                else:
                    think_parts.append(sample["model_response"][indices[i] + len("</think>"):indices[i + 1]].strip())
            
            
            think = " ".join(think_parts).strip()

            last_index = indices[-1] + len("</think>")
            answer = sample["model_response"][last_index:].strip()

            if len(think) != 0 and len(answer) != 0:
                think_success = -3
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) != 0 and len(answer) == 0:
                think_success = -4
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) == 0 and len(answer) != 0:
                think_success = -5
                test_print(sample, think, answer, think_success, use_error_print)
            else:
                think_success = -6
                test_print(sample, think, answer, think_success, use_error_print)



        elif sample["model_response"].count("</reason>") == 1:

            index = sample["model_response"].find("</reason>")
            think = sample["model_response"][:index].strip()

            index += len("</reason>")
            answer = sample["model_response"][index:].strip()

            if len(think) == 0 and len(answer) != 0:
                think_success = -1
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) != 0 and len(answer) == 0:
                think_success = -2
                test_print(sample, think, answer, think_success, use_error_print)
            else:
                think_success = 1
        
        elif sample["model_response"].count("</reason>") > 1:
            indices = []
            start = 0
            while True:
                index = sample["model_response"].find("</reason>", start)
                if index == -1:
                    break
                indices.append(index)
                start = index + len("</reason>")
            
            think_parts = []
            for i in range(len(indices) - 1):
                if i == 0:
                    think_parts.append(sample["model_response"][:indices[i]].strip())
                else:
                    think_parts.append(sample["model_response"][indices[i] + len("</reason>"):indices[i + 1]].strip())
            
            think = " ".join(think_parts).strip()

            last_index = indices[-1] + len("</reason>")
            answer = sample["model_response"][last_index:].strip()

            if len(think) != 0 and len(answer) != 0:
                think_success = -3
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) != 0 and len(answer) == 0:
                think_success = -4
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) == 0 and len(answer) != 0:
                think_success = -5
                test_print(sample, think, answer, think_success, use_error_print)
            else:
                think_success = -6
                test_print(sample, think, answer, think_success, use_error_print)



        elif sample["model_response"].count("</ reason>") == 1:

            index = sample["model_response"].find("</ reason>")
            think = sample["model_response"][:index].strip()

            index += len("</ reason>")
            answer = sample["model_response"][index:].strip()

            if len(think) == 0 and len(answer) != 0:
                think_success = -1
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) != 0 and len(answer) == 0:
                think_success = -2
                test_print(sample, think, answer, think_success, use_error_print)
            else:
                think_success = 1
        
        elif sample["model_response"].count("</ reason>") > 1:
            indices = []
            start = 0
            while True:
                index = sample["model_response"].find("</ reason>", start)
                if index == -1:
                    break
                indices.append(index)
                start = index + len("</ reason>")
            
            think_parts = []
            for i in range(len(indices) - 1):
                if i == 0:
                    think_parts.append(sample["model_response"][:indices[i]].strip())
                else:
                    think_parts.append(sample["model_response"][indices[i] + len("</ reason>"):indices[i + 1]].strip())
            
            
            think = " ".join(think_parts).strip()

            
            last_index = indices[-1] + len("</ reason>")
            answer = sample["model_response"][last_index:].strip()

            if len(think) != 0 and len(answer) != 0:
                think_success = -3
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) != 0 and len(answer) == 0:
                think_success = -4
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) == 0 and len(answer) != 0:
                think_success = -5
                test_print(sample, think, answer, think_success, use_error_print)
            else:
                think_success = -6
                test_print(sample, think, answer, think_success, use_error_print)



        elif sample["model_response"].count("<|im_start|>answer") == 1:

            index = sample["model_response"].find("<|im_start|>answer")
            think = sample["model_response"][:index].strip()

            index += len("<|im_start|>answer")
            answer = sample["model_response"][index:].strip()

            if len(think) == 0 and len(answer) != 0:
                think_success = -1
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) != 0 and len(answer) == 0:
                think_success = -2
                test_print(sample, think, answer, think_success, use_error_print)
            else:
                think_success = 1
        
        elif sample["model_response"].count("<|im_start|>answer") > 1:
            
            indices = []
            start = 0
            while True:
                index = sample["model_response"].find("<|im_start|>answer", start)
                if index == -1:
                    break
                indices.append(index)
                start = index + len("<|im_start|>answer")
            
            
            think_parts = []
            for i in range(len(indices) - 1):
                if i == 0:
                    think_parts.append(sample["model_response"][:indices[i]].strip())
                else:
                    think_parts.append(sample["model_response"][indices[i] + len("<|im_start|>answer"):indices[i + 1]].strip())
            
            
            think = " ".join(think_parts).strip()

            
            last_index = indices[-1] + len("<|im_start|>answer")
            answer = sample["model_response"][last_index:].strip()

            if len(think) != 0 and len(answer) != 0:
                think_success = -3
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) != 0 and len(answer) == 0:
                think_success = -4
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) == 0 and len(answer) != 0:
                think_success = -5
                test_print(sample, think, answer, think_success, use_error_print)
            else:
                think_success = -6
                test_print(sample, think, answer, think_success, use_error_print)


        elif sample["model_response"].count("</thought>") == 1:

            index = sample["model_response"].find("</thought>")
            think = sample["model_response"][:index].strip()

            index += len("</thought>")
            answer = sample["model_response"][index:].strip()

            if len(think) == 0 and len(answer) != 0:
                think_success = -1
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) != 0 and len(answer) == 0:
                think_success = -2
                test_print(sample, think, answer, think_success, use_error_print)
            else:
                think_success = 1
        
        elif sample["model_response"].count("</thought>") > 1:
            
            indices = []
            start = 0
            while True:
                index = sample["model_response"].find("</thought>", start)
                if index == -1:
                    break
                indices.append(index)
                start = index + len("</thought>")
            
            think_parts = []
            for i in range(len(indices) - 1):
                if i == 0:
                    think_parts.append(sample["model_response"][:indices[i]].strip())
                else:
                    think_parts.append(sample["model_response"][indices[i] + len("</thought>"):indices[i + 1]].strip())
            
            think = " ".join(think_parts).strip()

            last_index = indices[-1] + len("</thought>")
            answer = sample["model_response"][last_index:].strip()

            if len(think) != 0 and len(answer) != 0:
                think_success = -3
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) != 0 and len(answer) == 0:
                think_success = -4
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) == 0 and len(answer) != 0:
                think_success = -5
                test_print(sample, think, answer, think_success, use_error_print)
            else:
                think_success = -6
                test_print(sample, think, answer, think_success, use_error_print)




        elif len([marker for marker in special_markers if marker in sample["model_response"]]) >= 1:
            found_markers = [marker for marker in special_markers if marker in sample["model_response"]]
            last_marker = max(found_markers, key=lambda marker: sample["model_response"].rfind(marker))
            last_index = sample["model_response"].rfind(last_marker)
            
            think = sample["model_response"][:last_index].strip()
            answer = sample["model_response"][last_index + len(last_marker):].strip()
            
            for marker in found_markers:
                think = think.replace(marker, "").strip()
                answer = answer.replace(marker, "").strip()

            if len(think) != 0 and len(answer) != 0:
                think_success = -7
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) != 0 and len(answer) == 0:
                think_success = -8
                test_print(sample, think, answer, think_success, use_error_print)
            elif len(think) == 0 and len(answer) != 0:
                think_success = -9
                test_print(sample, think, answer, think_success, use_error_print)
            else:
                think_success = -6
                test_print(sample, think, answer, think_success, use_error_print)

        else:
            think = sample["model_response"]
            answer = ""
            think_success = -10
            test_print(sample, think, answer, think_success, use_error_print)
    else:
        think = ""
        answer = ""
        think_success = -11
        test_print(sample, think, answer, think_success, use_error_print)

    return think, answer, think_success


def split_main(file_path, save_path, model_name):
    data_list = read_json_to_list(file_path)
    
    false_1 = 0
    false_2 = 0
    false_3 = 0
    false_4 = 0
    false_5 = 0
    false_6 = 0
    false_7 = 0
    false_8 = 0
    false_9 = 0
    false_10 = 0
    false_11 = 0
    
    think_map = {
        -1: 101,
        -2: 200,
        -3: 201,
        -4: 202,
        -5: 203,
        -6: 300,
        -7: 204,
        -8: 205,
        -9: 206,
        -10: 207,
        -11: 301,
        1: 100,
    }
    
    # use_error_print = False
    use_error_print = True
    for sample in data_list:
        think, answer, think_success = judge_special_flag(sample, use_error_print)
        sample["thinking_part"] = think
        sample["answering_part"] = answer
        sample["thinking_part_tokens"] = count_tokens(think)
        sample["answering_part_tokens"] = count_tokens(answer)
        sample["think_success"] = think_map[think_success]

        if think_success == -1:
            false_1 += 1
        elif think_success == -2:
            false_2 += 1
        elif think_success == -3:
            false_3 += 1
        elif think_success == -4:
            false_4 += 1
        elif think_success == -5:
            false_5 += 1
        elif think_success == -6:
            false_6 += 1
        elif think_success == -7:
            false_7 += 1
        elif think_success == -8:
            false_8 += 1
        elif think_success == -9:
            false_9 += 1
        elif think_success == -10:
            false_10 += 1
        elif think_success == -11:
            false_11 += 1

    print(f"type 101: {false_1}")
    print(f"type 200: {false_2}")
    print(f"type 201: {false_3}")
    print(f"type 202: {false_4}")
    print(f"type 203: {false_5}")
    print(f"type 300: {false_6}")
    print(f"type 204: {false_7}")
    print(f"type 205: {false_8}")
    print(f"type 206: {false_9}")
    print(f"type 207: {false_10}")
    print(f"type 301: {false_11}")

    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=4)
    print(f"read response path in {file_path}")
    print(f"save split path in {save_path}")
    print(100*"~")
    print("\n\n")



if __name__ == "__main__":
    save_root_1 = base_directory+"/LRM_split"
    save_root_2 = base_directory+"/LRM_split_temp_0"
    k=5  # acc@k

    model_list = ["DS-R1-1.5B", "DS-R1-7B", "DS-R1-8B", "DS-R1-14B", "DS-R1-32B", "DS-R1-70B", "QwQ-32B", "L-R1-7B-DS", 
                    "L-R1-14B-DS", "L-R1-32B-DS", "L-R1-32B", "s1.1-7B", "s1.1-14B", "s1.1-32B", 
                    "EXAONE-2.4B", "EXAONE-7.8B", "EXAONE-32B", "Nemotron-8B", "Nemotron-49B", "Sky-T1-32B"]

    for model_name in model_list:
        print(f"top-p sampling setting, model is [ {model_name} ]")
        for i in range(k):
            file_path = base_directory + f"/LRM_local_vllm/{model_name}/LRM_response_{model_name}_{str(i)}.json"
            use_local = True
                
            save_path = save_root_1 + f"/{model_name}/LRM_response_split_{model_name}_{str(i)}.json"
            print(f"Starting new split. Results will be saved in {save_path}")
            split_main(file_path, save_path, model_name)


    for model_name in model_list:
        print(f"greedy setting, model is [ {model_name} ]")
        for i in range(1):
            file_path = base_directory + f"/LRM_local_vllm_temp_0/{model_name}/LRM_response_{model_name}_{str(i)}.json"
            use_local = True
                
            save_path = save_root_2 + f"/{model_name}/LRM_response_split_{model_name}_{str(i)}.json"
            print(f"Starting new split. Results will be saved in {save_path}")
            split_main(file_path, save_path, model_name)