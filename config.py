from re import M


MODEL_CONFIG = {
    # deepseek model is said to avoid use system prompt as in the document on https://huggingface.co/deepseek-ai/DeepSeek-R1
    "DS-R1-1.5B": {
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "DS-R1-7B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "DS-R1-8B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "DS-R1-14B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "DS-R1-32B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "DS-R1-70B":{
        "n_gpu": 4,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "L-R1-7B-DS":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "L-R1-14B-DS":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "L-R1-32B-DS":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "L-R1-32B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "s1.1-7B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "s1.1-14B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "s1.1-32B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "EXAONE-2.4B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "EXAONE-7.8B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "EXAONE-32B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "Nemotron-8B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt": "detailed thinking on",
        "generation_config":{}
    },
    "Nemotron-49B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt": "detailed thinking on",
        "generation_config":{}
    },
    "Sky-T1-32B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },

    # add your model here
    
}



