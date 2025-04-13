import pdb
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import re
from pprint import pprint
current_file_path = os.path.abspath(__file__)
base_directory = os.path.dirname(current_file_path)


def read_jsonl_to_list(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                result.append(data)
    return result



if __name__ == "__main__":
    # model_list = ["DS-R1-1.5B", "DS-R1-7B", "DS-R1-8B", "DS-R1-14B", "DS-R1-32B", "DS-R1-70B", "QwQ-32B", "L-R1-7B-DS", 
    #             "L-R1-14B-DS", "L-R1-32B-DS", "L-R1-32B", "s1.1-7B", "s1.1-14B", "s1.1-32B", 
    #             "EXAONE-2.4B", "EXAONE-7.8B", "EXAONE-32B", "Nemotron-8B", "Nemotron-49B", "Sky-T1-32B", "DS-R1", "Hunyuan-T1"]
    model_list = ["DS-R1-1.5B", "DS-R1"]
    model_results = {} 

    for model_name in model_list:
        model_stats = {
            "pass_acc_loose": 0,
            "pass_acc_strict": 0,
            "total_acc_loose": 0,
            "total_acc_strict": 0,
            "success_rate_loose": 0,
            "success_rate_strict": 0,
            "avg_tokens_strict": 0,
            "avg_tokens_loose": 0,
        }


        data_list_5 = []
        
        for i in range(5):
            file_path = os.path.join(base_directory, f"LRM_acc_eval/{model_name}/LRM_response_eval_{model_name}_{i}.json")
            data_list = read_jsonl_to_list(file_path)
            data_list_5.extend(data_list)

        # pdb.set_trace()

        loose_pass = 0
        loose_total = 0
        strict_pass = 0
        strict_total = 0
        total_think_token_strict = 0
        total_answer_token_strict = 0

        total_think_token_loose = 0
        total_answer_token_loose = 0

        success_num_loose = 0
        success_num_strict = 0


        for this_id in range(1, 423):
            total_flag_easy = 0
            total_flag_hard = 0

            for sample in data_list_5:
                if this_id == sample["ID"]:
                    
                    if sample["think_success"] in [100, 101]:
                        # strict
                        total_think_token_strict += sample["thinking_part_tokens"]
                        total_answer_token_strict += sample["answering_part_tokens"]
                        success_num_strict += 1
                        if sample["eval_result"] in [1]:
                            total_flag_hard += 1
                            strict_pass += 1

                    if sample["think_success"] not in [300, 301]:
                        # loose
                        total_think_token_loose += sample["thinking_part_tokens"]
                        total_answer_token_loose += sample["answering_part_tokens"]
                        success_num_loose += 1
                        if sample["eval_result"] in [1]:
                            total_flag_easy += 1
                            loose_pass += 1

            if total_flag_easy == 5:
                loose_total += 1
            if total_flag_hard == 5:
                strict_total += 1

        s1bench_len = 422
        model_stats["pass_acc_loose"] = loose_pass / (5*s1bench_len)
        model_stats["pass_acc_strict"] = strict_pass / (5*s1bench_len)
        model_stats["total_acc_loose"] = loose_total / s1bench_len
        model_stats["total_acc_strict"] = strict_total / s1bench_len
        model_stats["success_rate_loose"] = success_num_loose / (5*s1bench_len)
        model_stats["success_rate_strict"] = success_num_strict / (5*s1bench_len)
        model_stats["avg_tokens_strict"] = (total_think_token_strict+total_answer_token_strict) / success_num_strict
        model_stats["avg_tokens_loose"] = (total_think_token_loose+total_answer_token_loose) / success_num_loose

        model_results[model_name] = model_stats
    
    pprint(model_results, indent=4)
    # pdb.set_trace()
    





    



