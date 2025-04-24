from dataclasses import dataclass, fields
import gradio as gr
import redis
import threading
import time
import os
import requests
import json
import subprocess
import sys
import ast
import time
from datetime import datetime
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import huggingface_hub
from huggingface_hub import snapshot_download
import logging
import psutil
import git
from git import Repo




REQUEST_TIMEOUT = 300
def wait_for_backend(backend_url, timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.post(backend_url, json={"method": "list"}, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                print("Backend container is online.")
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass  # Backend is not yet reachable
        time.sleep(5)  # Wait for 5 seconds before retrying
    print(f"Timeout: Backend container did not come online within {timeout} seconds.")
    return False


test_vllms = []
test_vllms_list_running = []

docker_container_list = []
current_models_data = []
db_gpu_data = []
db_gpu_data_len = ''
SELECTED_MODEL_ID = ''
MEM_TOTAL = 0
MEM_USED = 0
MEM_FREE = 0
PROMPT = "A famous quote"
SEARCH_INPUT_TS = 0
SEARCH_INPUT_THRESHOLD = 10
SEARCH_REQUEST_TIMEOUT = 3
SEARCH_INITIAL_DELAY = 10
DOCKER_API_URL = f'http://container_backend:{os.getenv("BACKEND_PORT")}/docker'
REDIS_DB_VLLM = "db_test28"
REDIS_API_URL = f'http://container_backend:{os.getenv("BACKEND_PORT")}/redis'

error_vllm = {
    "container_name": "error",
    "uid": "0000000000",
    "status": "offline",
    "State": {
        "Status": "offline"
    },
    "gpu": {
        "mem": "nuh uh 0%"
    },
    "ts": "0"
}

error_vllm2 = {
    "container_name": "error2",
    "uid": "1111111111111",
    "status": "offline",
    "State": {
        "Status": "offline"
    },
    "gpu": {
        "mem": "nuh uh 1%"
    },
    "ts": "0"
}


error_vllm3 = {
    "container_name": "error3",
    "uid": "222222222222",
    "status": "offline",
    "State": {
        "Status": "offline"
    },
    "gpu": {
        "mem": "nuh uh 2%"
    },
    "ts": "0"
}


error_vllm4 = {
    "container_name": "error4",
    "uid": "3333333333",
    "status": "offline",
    "State": {
        "Status": "offline"
    },
    "gpu": {
        "mem": "nuh uh 3%"
    },
    "ts": "0"
}


error_vllm5 = {
    "container_name": "error5",
    "uid": "4444444444",
    "status": "offline",
    "State": {
        "Status": "offline"
    },
    "gpu": {
        "mem": "nuh uh 4%"
    },
    "ts": "0"
}


try:
    r = redis.Redis(host="redis", port=6379, db=0)
    db_gpu = json.loads(r.get('db_gpu'))
    # print(f'db_gpu: {db_gpu} {len(db_gpu)}')
    db_gpu_data_len = len(db_gpu_data)
except Exception as e:
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')






def redis_api(*req_component,**req_dict):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] %%%%%%%%%%%% [redis] START')
    global res_vllms
    # put in default later
    error_vllm = {
        "container_name": "error",
        "uid": "0000000000",
        "status": "offline",
        "State": {
            "Status": "offline"
        },
        "gpu": {
            "mem": "nuh uh%"
        },
        "ts": "0"
    }
    try:
        global REDIS_API_URL
        
        if not req_dict:
            print(f' %%%%%%%%%%%% [redis_api]: Error: no req_dict')
            return [error_vllm]
        else:
            print(f' %%%%%%%%%%%% [redis_api]: req_dict: {req_dict}')
            print(f' %%%%%%%%%%%% [redis_api]: req_dict["req_dict"]: {req_dict["req_dict"]}')
            print(f' %%%%%%%%%%%% [redis_api]: req_dict["req_dict"]["method"]: {req_dict["req_dict"]["method"]}')
        
        print(f' %%%%%%%%%%%% [redis_api]: req_component: {req_component}')
        
        if not req_dict["req_dict"]["method"]:
            print(f' %%%%%%%%%%%% [redis_api]: Error: no method')
            return [error_vllm]
        
        
        print(f' %%%%%%%%%%%% [redis_api]: VOR REQUESTINNG')
        
        if req_dict["req_dict"]["method"] == "test":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] %%%%%%%%%%%% [redis] ["method"] == "test"')
            
            response = requests.post(REDIS_API_URL, json={"method":req_dict["req_dict"]["method"]})
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] %%%%%%%%%%%% [redis] response: {response}')
            
            res_json = response.json()
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] %%%%%%%%%%%% [redis] res_json: {res_json}')
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] %%%%%%%%%%%% [redis] res_json["result_status"]: {res_json["result_status"]}')
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] %%%%%%%%%%%% [redis] res_json["result_data"]: {res_json["result_data"]}')

            return res_json["result_data"]
        
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [redis_api] {e}')
        return [error_vllm]





backend_url = f'http://container_backend:{os.getenv("BACKEND_PORT")}/docker'
print(f' ~~~~~~~~~~~~~~~~~~~~~~~~ FRONTEND WAITING FOR BACKEND BOOT TO GET VLLMS')
print(f' ~~~~~~~~~~~~~~~~~~~~~~~~ querying ...')
res_backend = wait_for_backend(backend_url)
print(f' ~~~~~~~~~~~~~~~~~~~~~~~~ true or no?  res_backend: {res_backend}')

res_vllms = []
if wait_for_backend(backend_url):
    print(f' ~~~~~~~~~~~~~~~~~~~~~~~~ ok is true... trying to get vllms ...')

    req_test = {
        "method": "test"
    }

    
    print(f' ~~~~~~~~~~~~~~~~~~~~~~~~ 1')
    res_vllms = redis_api("test", req_dict=req_test)
    print(f' ~~~~~~~~~~~~~~~~~~~~~~~~ 2')
    print(f' ~~~~~~~~~~~~~~~~~~~~~~~~ OK GOT res_vllms: {res_vllms}')
    print(f' ~~~~~~~~~~~~~~~~~~~~~~~~ OK GOT res_vllms[0]: {res_vllms[0]}')
    print(f' ~~~~~~~~~~~~~~~~~~~~~~~~ OK GOT res_vllms[1]: {res_vllms[1]}')

    print(f' ~~~~~~~~~~~~~~~~~~~~~~~~ 3')



else:    
    print(f' ~~~~~~~~~~~ ERRROR ~~~~~~~~~~~~~ 4 responded False')













LOG_PATH= './logs'
LOGFILE_CONTAINER = './logs/logfile_container_frontend.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.INFO, format='[%(asctime)s - %(name)s - %(levelname)s - %(message)s]')
logging.info(f' [START] started logging in {LOGFILE_CONTAINER}')

def load_log_file(req_container_name):
    print(f' **************** GOT LOG FILE REQUEST FOR CONTAINER ID: {req_container_name}')
    logging.info(f' **************** GOT LOG FILE REQUEST FOR CONTAINER ID: {req_container_name}')
    try:
        with open(f'{LOG_PATH}/logfile_{req_container_name}.log', "r", encoding="utf-8") as file:
            lines = file.readlines()
            last_20_lines = lines[-20:]
            reversed_lines = last_20_lines[::-1]
            return ''.join(reversed_lines)
    except Exception as e:
        return f'{e}'



DEFAULTS_PATH = "/usr/src/app/utils/defaults.json"
if not os.path.exists(DEFAULTS_PATH):
    logging.info(f' [START] File missing: {DEFAULTS_PATH}')

with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
    defaults_frontend = json.load(f)["frontend"]
    logging.info(f' [START] SUCCESS! Loaded: {DEFAULTS_PATH}')
    logging.info(f' [START] {len(defaults_frontend['vllm_supported_architectures'])} supported vLLM architectures found!')



























        
        
def dropdown_load_tested_models():

    global current_models_data
    response_models = defaults_frontend['tested_models']
    print(f'response_models: {response_models}')
    current_models_data = response_models.copy()
    model_ids = [m["id"] for m in response_models]
    print(f'model_ids: {model_ids}')
    # return gr.update(choices=model_ids, value=response_models[0]["id"], visible=True)
    return [gr.update(choices=model_ids, value=response_models[0]["id"], visible=True),gr.update(value=response_models[0]["id"],show_label=True, label=f'Loaded {len(model_ids)} models!')]



def get_network_data():
    try:
        res_network_data_all = json.loads(r.get('db_network'))
        return res_network_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e

def get_gpu_data():
    try:
        res_gpu_data_all = json.loads(r.get('db_gpu'))
        return res_gpu_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e

def get_disk_data():
    try:
        res_disk_data_all = json.loads(r.get('db_disk'))
        return res_disk_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e





def docker_api(req_method,req_var):
    try:
        global DOCKER_API_URL
        global docker_container_list
        
        if req_method == "list":
            response = requests.post(DOCKER_API_URL, json={"method":req_method})
            res_json = response.json()
            docker_container_list = res_json.copy()
            if response.status_code == 200:
                return res_json
            else:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return f'Error: {response.status_code}'

        if req_method == "logs":
            response = requests.post(DOCKER_API_URL, json={"method":req_method,"model":req_var})
            res_json = response.json()
            return ''.join(res_json["result_data"])
        
        if req_method == "start":
            response = requests.post(DOCKER_API_URL, json={"method":req_method,"model":req_var})
            res_json = response.json()
            return res_json
        
        if req_method == "stop":
            response = requests.post(DOCKER_API_URL, json={"method":req_method,"model":req_var})
            res_json = response.json()
            return res_json
        
        if req_method == "delete":
            response = requests.post(DOCKER_API_URL, json={"method":req_method,"model":req_var})
            res_json = response.json()
            return res_json
        
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker_api] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker_api] {e}'




















































def search_change(input_text):
    global SEARCH_INPUT_TS
    global current_models_data
    current_ts = int(datetime.now().timestamp())
    if SEARCH_INPUT_TS + SEARCH_INPUT_THRESHOLD > current_ts:
        wait_time = SEARCH_INPUT_TS + SEARCH_INPUT_THRESHOLD - current_ts
        return [gr.update(show_label=False),gr.update(show_label=True, label=f'Found {len(current_models_data)} models! Please wait {wait_time} sec or click on search')]
    if len(input_text) < 3: 
        # return [gr.update(show_label=False),gr.update(show_label=True, label=" < 3")]
        return [gr.update(show_label=False),gr.update(show_label=True)]
    if SEARCH_INPUT_TS == 0 and len(input_text) > 5:
        SEARCH_INPUT_TS = int(datetime.now().timestamp())
        res_huggingface_hub_search_model_ids,  res_huggingface_hub_search_current_value = search_models(input_text)
        if len(res_huggingface_hub_search_model_ids) >= 1000:
            return [gr.update(choices=res_huggingface_hub_search_model_ids, value=res_huggingface_hub_search_current_value, visible=True),gr.update(show_label=True, label=f'Found >1000 models!')]
        return [gr.update(choices=res_huggingface_hub_search_model_ids, value=res_huggingface_hub_search_current_value, visible=True),gr.update(show_label=True, label=f'Found {len(res_huggingface_hub_search_model_ids)} models!')]
        
    if SEARCH_INPUT_TS == 0:
        SEARCH_INPUT_TS = int(datetime.now().timestamp()) + SEARCH_INITIAL_DELAY
        return [gr.update(show_label=False),gr.update(show_label=True, label=f'Waiting auto search {SEARCH_INITIAL_DELAY} sec')]
    if SEARCH_INPUT_TS + SEARCH_INPUT_THRESHOLD <= current_ts:
        SEARCH_INPUT_TS = int(datetime.now().timestamp())
        res_huggingface_hub_search_model_ids,  res_huggingface_hub_search_current_value = search_models(input_text)
        if len(res_huggingface_hub_search_model_ids) >= 1000:
            return [gr.update(choices=res_huggingface_hub_search_model_ids, value=res_huggingface_hub_search_current_value, visible=True),gr.update(show_label=True, label=f'Found >1000 models!')]
        return [gr.update(choices=res_huggingface_hub_search_model_ids, value=res_huggingface_hub_search_current_value, visible=True),gr.update(show_label=True, label=f'Found {len(res_huggingface_hub_search_model_ids)} models!')]















def search_models(query):
    try:
        global current_models_data    
        response = requests.get(f'https://huggingface.co/api/models?search={query}')
        response_models = response.json()
        current_models_data = response_models.copy()
        model_ids = [m["id"] for m in response_models]
        if len(model_ids) < 1:
            model_ids = ["No models found!"]
        return gr.update(choices=model_ids, value=response_models[0]["id"], show_label=True, label=f'found {len(response_models)} models!')
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')























def format_bytes(req_format, req_size):
    if req_format == "human":
        for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
            if abs(req_size) < 1024.0:
                return f'{req_size:3.1f}{unit}B'
            req_size /= 1024.0
        return f'{req_size:.1f}YiB'
    elif req_format == "bytes":
        req_size = req_size.upper()
        if 'KB' in req_size:
            return int(float(req_size.replace('KB', '').strip()) * 1024)
        elif 'MB' in req_size:
            return int(float(req_size.replace('MB', '').strip()) * 1024 * 1024)
        elif 'GB' in req_size:
            return int(float(req_size.replace('GB', '').strip()) * 1024 * 1024 * 1024)
        elif 'B' in req_size:
            return int(float(req_size.replace('B', '').strip()))
        return 0
    else:
        raise ValueError("Invalid format specified. Use 'human' or 'bytes'.")




def convert_to_bytes(size_str):
    """Convert human-readable file size to bytes"""
    size_str = size_str.upper()
    if 'KB' in size_str:
        return int(float(size_str.replace('KB', '').strip()) * 1024)
    elif 'MB' in size_str:
        return int(float(size_str.replace('MB', '').strip()) * 1024 * 1024)
    elif 'GB' in size_str:
        return int(float(size_str.replace('GB', '').strip()) * 1024 * 1024 * 1024)
    elif 'B' in size_str:
        return int(float(size_str.replace('B', '').strip()))
    return 0








def get_git_model_size(selected_id):    
    try:
        repo = Repo.clone_from(f'https://huggingface.co/{selected_id}', selected_id, no_checkout=True)
    except git.exc.GitCommandError as e:
        if "already exists and is not an empty directory" in str(e):
            repo = Repo(selected_id)
        else:
            raise
    
    lfs_files = repo.git.lfs("ls-files", "-s").splitlines()
    files_list = []
    for line in lfs_files:
        parts = line.split(" - ")
        if len(parts) == 2:
            file_hash, file_info = parts
            file_parts = file_info.rsplit(" (", 1)
            if len(file_parts) == 2:
                file_name = file_parts[0]
                size_str = file_parts[1].replace(")", "")
                size_bytes = format_bytes("bytes",size_str)
                
                files_list.append({
                    "id": file_hash.strip(),
                    "file": file_name.strip(),
                    "size": size_bytes,
                    "size_human": size_str
                })
            
        
    return sum([file["size"] for file in files_list]), format_bytes("human",sum([file["size"] for file in files_list]))
    
















def calculate_model_size(json_info): # to fix    
    try:
        d_model = json_info.get("hidden_size") or json_info.get("d_model")
        num_hidden_layers = json_info.get("num_hidden_layers", 0)
        num_attention_heads = json_info.get("num_attention_heads") or json_info.get("decoder_attention_heads") or json_info.get("encoder_attention_heads", 0)
        intermediate_size = json_info.get("intermediate_size") or json_info.get("encoder_ffn_dim") or json_info.get("decoder_ffn_dim", 0)
        vocab_size = json_info.get("vocab_size", 0)
        num_channels = json_info.get("num_channels", 3)
        patch_size = json_info.get("patch_size", 16)
        torch_dtype = json_info.get("torch_dtype", "float32")
        bytes_per_param = 2 if torch_dtype == "float16" else 4
        total_size_in_bytes = 0
        
        if json_info.get("model_type") == "vit":
            embedding_size = num_channels * patch_size * patch_size * d_model
            total_size_in_bytes += embedding_size

        if vocab_size and d_model:
            embedding_size = vocab_size * d_model
            total_size_in_bytes += embedding_size

        if num_attention_heads and d_model and intermediate_size:
            attention_weights_size = num_hidden_layers * (d_model * d_model * 3)
            ffn_weights_size = num_hidden_layers * (d_model * intermediate_size + intermediate_size * d_model)
            layer_norm_weights_size = num_hidden_layers * (2 * d_model)

            total_size_in_bytes += (attention_weights_size + ffn_weights_size + layer_norm_weights_size)

        if json_info.get("is_encoder_decoder"):
            encoder_size = num_hidden_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            decoder_layers = json_info.get("decoder_layers", 0)
            decoder_size = decoder_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            
            total_size_in_bytes += (encoder_size + decoder_size)

        return total_size_in_bytes * 2
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return 0





def get_info(selected_id):
    
    print(f' @@@ [get_info] 0')
    print(f' @@@ [get_info] 0')   
    container_name = ""
    res_model_data = {
        "search_data" : "",
        "model_id" : "",
        "pipeline_tag" : "",
        "architectures" : "",
        "transformers" : "",
        "private" : "",
        "downloads" : ""
    }
    
    if selected_id == None:
        print(f' @@@ [get_info] selected_id NOT FOUND!! RETURN ')
        print(f' @@@ [get_info] selected_id NOT FOUND!! RETURN ') 
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
    
    global CURRENT_MODELS_DATA
    global SELECTED_MODEL_ID
    SELECTED_MODEL_ID = selected_id
    print(f' @@@ [get_info] {selected_id} 2')
    print(f' @@@ [get_info] {selected_id} 2')  
    
    print(f' @@@ [get_info] {selected_id} 3')
    print(f' @@@ [get_info] {selected_id} 3')  
    container_name = str(res_model_data["model_id"]).replace('/', '_')
    print(f' @@@ [get_info] {selected_id} 4')
    print(f' @@@ [get_info] {selected_id} 4')  
    if len(CURRENT_MODELS_DATA) < 1:
        print(f' @@@ [get_info] len(CURRENT_MODELS_DATA) < 1! RETURN ')
        print(f' @@@ [get_info] len(CURRENT_MODELS_DATA) < 1! RETURN ') 
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
    try:
        print(f' @@@ [get_info] {selected_id} 5')
        print(f' @@@ [get_info] {selected_id} 5') 
        for item in CURRENT_MODELS_DATA:
            print(f' @@@ [get_info] {selected_id} 6')
            print(f' @@@ [get_info] {selected_id} 6') 
            if item['id'] == selected_id:
                print(f' @@@ [get_info] {selected_id} 7')
                print(f' @@@ [get_info] {selected_id} 7') 
                res_model_data["search_data"] = item
                
                if "pipeline_tag" in item:
                    res_model_data["pipeline_tag"] = item["pipeline_tag"]
  
                if "tags" in item:
                    if "transformers" in item["tags"]:
                        res_model_data["transformers"] = True
                    else:
                        res_model_data["transformers"] = False
                                    
                if "private" in item:
                    res_model_data["private"] = item["private"]
                                  
                if "architectures" in item:
                    res_model_data["architectures"] = item["architectures"][0]
                                                    
                if "downloads" in item:
                    res_model_data["downloads"] = item["downloads"]
                  
                container_name = str(res_model_data["model_id"]).replace('/', '_')
                
                print(f' @@@ [get_info] {selected_id} 8')
                print(f' @@@ [get_info] {selected_id} 8') 
                
                return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
            else:
                
                print(f' @@@ [get_info] {selected_id} 9')
                print(f' @@@ [get_info] {selected_id} 9') 
                
                return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
    except Exception as e:
        print(f' @@@ [get_info] {selected_id} 10')
        print(f' @@@ [get_info] {selected_id} 10') 
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name








def get_additional_info(selected_id):    
        res_model_data = {
            "hf_data" : "",
            "hf_data_config" : "",
            "config_data" : "",
            "architectures" : "",
            "model_type" : "",
            "quantization" : "",
            "tokenizer_config" : "",
            "model_id" : selected_id,
            "size" : 0,
            "size_human" : 0,
            "gated" : "",
            "torch_dtype" : "",
            "hidden_size" : "",
            "cuda_support" : "",
            "compute_capability" : ""
        }                
        try:
            try:
                model_info = huggingface_hub.model_info(selected_id)
                model_info_json = vars(model_info)
                res_model_data["hf_data"] = model_info_json
                
                if "config" in model_info.__dict__:
                    res_model_data['hf_data_config'] = model_info_json["config"]
                    if "architectures" in model_info_json["config"]:
                        res_model_data['architectures'] = model_info_json["config"]["architectures"][0]
                    if "model_type" in model_info_json["config"]:
                        res_model_data['model_type'] = model_info_json["config"]["model_type"]
                    if "tokenizer_config" in model_info_json["config"]:
                        res_model_data['tokenizer_config'] = model_info_json["config"]["tokenizer_config"]
                               
                if "gated" in model_info.__dict__:
                    res_model_data['gated'] = model_info_json["gated"]
                
                if "safetensors" in model_info.__dict__:
                    print(f'  FOUND safetensors')
                    print(f'  GFOUND safetensors')   
                    
                    safetensors_json = vars(model_info.safetensors)
                    
                    
                    print(f'  FOUND safetensors:::::::: {safetensors_json}')
                    print(f'  GFOUND safetensors:::::::: {safetensors_json}') 
                    try:
                        quantization_key = next(iter(safetensors_json['parameters'].keys()))
                        print(f'  FOUND first key in parameters:::::::: {quantization_key}')
                        res_model_data['quantization'] = quantization_key
                        
                    except Exception as get_model_info_err:
                        print(f'  first key NOT FOUND in parameters:::::::: {quantization_key}')
                        pass
                    
                    print(f'  FOUND safetensors TOTAL :::::::: {safetensors_json["total"]}')
                    print(f'  GFOUND safetensors:::::::: {safetensors_json["total"]}')
                                        
                    print(f'  ooOOOOOOOOoooooo res_model_data["quantization"] {res_model_data["quantization"]}')
                    print(f'ooOOOOOOOOoooooo res_model_data["quantization"] {res_model_data["quantization"]}')
                    if res_model_data["quantization"] == "F32":
                        print(f'  ooOOOOOOOOoooooo found F32 -> x4')
                        print(f'ooOOOOOOOOoooooo found F32 -> x4')
                    else:
                        print(f'  ooOOOOOOOOoooooo NUUUH FIND F32 -> x2')
                        print(f'ooOOOOOOOOoooooo NUUUH FIND F32 -> x2')
                        res_model_data['size'] = int(safetensors_json["total"]) * 2
                else:
                    print(f' !!!!DIDNT FIND safetensors !!!! :::::::: ')
                    print(f' !!!!!! DIDNT FIND safetensors !!:::::::: ') 
            
            
            
            except Exception as get_model_info_err:
                res_model_data['hf_data'] = f'{get_model_info_err}'
                pass
                    
            try:
                response = requests.get(f'https://huggingface.co/{selected_id}/resolve/main/config.json', timeout=SEARCH_REQUEST_TIMEOUT)
                if response.status_code == 200:
                    response_json = response.json()
                    res_model_data["config_data"] = response_json
                    
                    if "architectures" in res_model_data["config_data"]:
                        res_model_data["architectures"] = res_model_data["config_data"]["architectures"][0]
                        
                    if "torch_dtype" in res_model_data["config_data"]:
                        res_model_data["torch_dtype"] = res_model_data["config_data"]["torch_dtype"]
                        print(f'  ooOOOOOOOOoooooo torch_dtype: {res_model_data["torch_dtype"]}')
                        print(f'ooOOOOOOOOoooooo torch_dtype: {res_model_data["torch_dtype"]}')
                    if "hidden_size" in res_model_data["config_data"]:
                        res_model_data["hidden_size"] = res_model_data["config_data"]["hidden_size"]
                        print(f'  ooOOOOOOOOoooooo hidden_size: {res_model_data["hidden_size"]}')
                        print(f'ooOOOOOOOOoooooo hidden_size: {res_model_data["hidden_size"]}')
                else:
                    res_model_data["config_data"] = f'{response.status_code}'
                    
            except Exception as get_config_json_err:
                res_model_data["config_data"] = f'{get_config_json_err}'
                pass                       
            
            

            res_model_data["size"], res_model_data["size_human"] = get_git_model_size(selected_id)
            
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["architectures"], res_model_data["model_id"], gr.update(value=res_model_data["size"], label=f'size ({res_model_data["size_human"]})'), res_model_data["gated"], res_model_data["model_type"], res_model_data["quantization"], res_model_data["torch_dtype"], res_model_data["hidden_size"]
        
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["model_id"], gr.update(value=res_model_data["size"], label=f'size ({res_model_data["size_human"]})'), res_model_data["gated"], res_model_data["model_type"],  res_model_data["quantization"], res_model_data["torch_dtype"], res_model_data["hidden_size"]


def gr_load_check(selected_model_id, selected_model_architectures, selected_model_pipeline_tag, selected_model_transformers, selected_model_size, selected_model_private, selected_model_gated, selected_model_model_type, selected_model_quantization):
    

    
    # check CUDA support mit backend call
    
    # if "gguf" in selected_model_id.lower():
    #     return f'Selected a GGUF model!', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    req_model_storage = "/models"
    req_model_path = f'{req_model_storage}/{selected_model_id}'
    
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] **************** [gr_load_check] searching {selected_model_id} in {req_model_storage} (req_model_path: {req_model_path}) ...')
    print(f' **************** [gr_load_check] searching {selected_model_id} in {req_model_storage} (req_model_path: {req_model_path})...')
    


    models_found = []
    # try:                   
    #     if os.path.isdir(req_model_storage):
    #         print(f' **************** found model storage path! {req_model_storage}')
    #         print(f' **************** getting folder elements ...')       
    #         print(f' **************** found model storage path! {req_model_storage}')
    #         print(f' **************** getting folder elements ...')                        
    #         for m_entry in os.listdir(req_model_storage):
    #             m_path = os.path.join(req_model_storage, m_entry)
    #             if os.path.isdir(m_path):
    #                 for item_sub in os.listdir(m_path):
    #                     sub_item_path = os.path.join(m_path, item_sub)
    #                     models_found.append(sub_item_path)        
    #         print(f' **************** found models ({len(models_found)}): {models_found}')
    #         print(f' **************** found models ({len(models_found)}): {models_found}')
    #     else:
    #         print(f' **************** found models ({len(models_found)}): {models_found}')

    # except Exception as e:
    #     print(f' **************** ERR getting models in {req_model_storage}: {e}')


    model_path = selected_model_id
    if req_model_path in models_found:
        print(f' **************** FOUND MODELS ALREADY!!! {selected_model_id} ist in {models_found}')
        model_path = req_model_path
        return f'Model already downloaded!', gr.update(visible=True), gr.update(visible=True)
    else:
        print(f' **************** NUH UH DIDNT FIND MODEL YET!! {selected_model_id} ist NAWT in {models_found}')
    
    
        
    if selected_model_architectures == '':
        return f'Selected model has no architecture', gr.update(visible=False), gr.update(visible=False)


    # if selected_model_architectures.lower() not in defaults_frontend['vllm_supported_architectures']:
    #     if selected_model_transformers != 'True':   
    #         return f'Selected model architecture is not supported by vLLM but transformers are available (you may try to load the model in gradio Interface)', gr.update(visible=True), gr.update(visible=True)
    #     else:
    #         return f'Selected model architecture is not supported by vLLM and has no transformers', gr.update(visible=False), gr.update(visible=False)     
    
    if selected_model_pipeline_tag == '':
        return f'Selected model has no pipeline tag', gr.update(visible=True), gr.update(visible=True)
            
    if selected_model_pipeline_tag not in ["text-generation","automatic-speech-recognition"]:
        return f'Only "text-generation" and "automatic-speech-recognition" models supported', gr.update(visible=False), gr.update(visible=False)
    
    if selected_model_private != 'False':        
        return f'Selected model is private', gr.update(visible=False), gr.update(visible=False)
        
    if selected_model_gated != 'False':        
        return f'Selected model is gated', gr.update(visible=False), gr.update(visible=False)
        
    if selected_model_transformers != 'True':        
        return f'Selected model has no transformers', gr.update(visible=True), gr.update(visible=True)
        
    if selected_model_size == '0':        
        return f'Selected model has no size', gr.update(visible=False), gr.update(visible=False)


    return f'Selected model is supported by vLLM!'







                
def toggle_compute_type(device):
    
    if device == 'cpu':
        return gr.update(choices=["int8"], value="int8")
    
    return gr.update(choices=["int8_float16", "float16"], value="float16")






def get_audio_path(audio_file):
    req_file = audio_file
    return [f'req_file: {req_file}', f'{req_file}']

def transcribe_audio(audio_model,audio_path,device,compute_type):  
    try:
        print(f'[transcribe_audio] audio_path ... {audio_path}')
        logging.info(f'[transcribe_audio] audio_path ... {audio_path}')
      
        AUDIO_URL = f'http://container_audio:{os.getenv("AUDIO_PORT")}/t'

        print(f'[transcribe_audio] AUDIO_URL ... {AUDIO_URL}')
        logging.info(f'[transcribe_audio] AUDIO_URL ... {AUDIO_URL}')

        print(f'[transcribe_audio] getting status ... ')
        logging.info(f'[transcribe_audio] getting status ... ')
        
        response = requests.post(AUDIO_URL, json={
            "method": "status"
        }, timeout=SEARCH_REQUEST_TIMEOUT)

        if response.status_code == 200:          
            print(f'[transcribe_audio] >> got response == 200 ... building json ... {response}')
            logging.info(f'[transcribe_audio] >> got response == 200 ... building json ... {response}')
            res_json = response.json()    
            print(f'[transcribe_audio] >> got res_json ... {res_json}')
            logging.info(f'[transcribe_audio] >> got res_json ... {res_json}')

            if res_json["result_data"] == "ok":
                print(f'[transcribe_audio] >> status: "ok" ... starting transcribe .... ')
                logging.info(f'[transcribe_audio] >> status: "ok" ... starting transcribe .... ')
      
                response = requests.post(AUDIO_URL, json={
                    "method": "transcribe",
                    "audio_model": audio_model,
                    "audio_path": audio_path,
                    "device": device,
                    "compute_type": compute_type
                })

                print(f'[transcribe_audio] >> got response #22222 == 200 ... building json ... {response}')
                logging.info(f'[transcribe_audio] >> got response #22222 == 200 ... building json ... {response}')
                
                res_json = response.json()
   
                print(f'[transcribe_audio] >> #22222 got res_json ... {res_json}')
                logging.info(f'[transcribe_audio] >> #22222 got res_json ... {res_json}')
                
                if res_json["result_status"] == 200:
                    return f'{res_json["result_data"]}'
                else: 
                    return 'Error :/'
            else:
                print('[transcribe_audio] ERROR AUDIO SERVER DOWN!?')
                logging.info('[transcribe_audio] ERROR AUDIO SERVER DOWN!?')
                return 'Error :/'

    except Exception as e:
        return f'Error: {e}'






























def network_to_pd():       
    rows = []
    try:
        network_list = get_network_data()
        # logging.info(f'[network_to_pd] network_list: {network_list}')  # Use logging.info instead of logging.exception
        for entry in network_list:

            rows.append({
                "container": entry["container"],
                "current_dl": entry["current_dl"]
            })
            
            
        df = pd.DataFrame(rows)
        return df,rows[0]["current_dl"]
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        rows.append({
                "container": "0",
                "current_dl": f'0',
                "timestamp": f'0',
                "info": f'0'
        })
        df = pd.DataFrame(rows)
        return df



def disk_to_pd():
    rows = []
    try:
        disk_list = get_disk_data()
        for entry in disk_list:
            disk_info = ast.literal_eval(entry['disk_info'])
            rows.append({                
                "disk_i": entry.get("disk_i", "0"),
                "timestamp": entry.get("timestamp", "0"),
                "device": disk_info.get("device", "0"),
                "usage_percent": disk_info.get("usage_percent", "0"),
                "mountpoint": disk_info.get("mountpoint", "0"),
                "fstype": disk_info.get("fstype", "0"),
                "opts": disk_info.get("opts", "0"),
                "usage_total": disk_info.get("usage_total", "0"),
                "usage_used": disk_info.get("usage_used", "0"),
                "usage_free": disk_info.get("usage_free", "0"),                
                "io_read_count": disk_info.get("io_read_count", "0"),
                "io_write_count": disk_info.get("io_write_count", "0")                
            })
        df = pd.DataFrame(rows)
        return df
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        logging.info(f' &&&&&& [ERROR] [disk_to_pd] GOT e {e}')

disk_to_pd()


def gpu_to_pd():
    global MEM_TOTAL
    global MEM_USED
    global MEM_FREE
    rows = []

    try:
        gpu_list = get_gpu_data()
        MEM_TOTAL = 0
        MEM_USED = 0
        MEM_FREE = 0
        for entry in gpu_list:
            gpu_info = ast.literal_eval(entry['gpu_info'])
            
            current_gpu_mem_total = gpu_info.get("mem_total", "0")
            current_gpu_mem_used = gpu_info.get("mem_used", "0")
            current_gpu_mem_free = gpu_info.get("mem_free", "0")
            MEM_TOTAL = float(MEM_TOTAL) + float(current_gpu_mem_total.split()[0])
            MEM_USED = float(MEM_USED) + float(current_gpu_mem_used.split()[0])
            MEM_FREE = float(MEM_FREE) + float(current_gpu_mem_free.split()[0])
            print(f'MMMMMMMMMMMMMMMMM MEM 1')
            update_mem(gpu_info.get("mem_util", "0"))
            print(f'MMMMMMMMMMMMMMMMM MEM 2')
            # mmmmmmmm
            rows.append({                                
                "name": gpu_info.get("name", "0"),
                "mem_util": gpu_info.get("mem_util", "0"),
                "timestamp": entry.get("timestamp", "0"),
                "fan_speed": gpu_info.get("fan_speed", "0"),
                "temperature": gpu_info.get("temperature", "0"),
                "gpu_util": gpu_info.get("gpu_util", "0"),
                "power_usage": gpu_info.get("power_usage", "0"),
                "clock_info_graphics": gpu_info.get("clock_info_graphics", "0"),
                "clock_info_mem": gpu_info.get("clock_info_mem", "0"),                
                "cuda_cores": gpu_info.get("cuda_cores", "0"),
                "compute_capability": gpu_info.get("compute_capability", "0"),
                "current_uuid": gpu_info.get("current_uuid", "0"),
                "gpu_i": entry.get("gpu_i", "0"),
                "supported": gpu_info.get("supported", "0"),
                "not_supported": gpu_info.get("not_supported", "0"),
                "status": "ok"
            })

        df = pd.DataFrame(rows)
        return df
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')

gpu_to_pd()
























def redis_connection(**kwargs):
    try:
        if not kwargs:
            print(f' **REDIS: Error: no kwargs')
            return False
        # else:
        #     print(f' **REDIS: kwargs: {kwargs}')
        
        if not kwargs["db_name"]:
            print(f' **REDIS: Error: no db_name')
            return False
            
        if not kwargs["method"]:
            print(f' **REDIS: Error: no method')
            return False
            
        if not kwargs["select"]:
            print(f' **REDIS: Error: no select')
            return False

        res_db_list = r.lrange(kwargs["db_name"], 0, -1)

        # print(f' **REDIS: found {len(res_db_list)} entries!')
        res_db_list = [json.loads(entry) for entry in res_db_list]
        
        if kwargs["select"] == "filter":
            if not kwargs["filter_key"]:
                print(f' **REDIS: Error: no filter_key')
                return False
            
            if not kwargs["filter_val"]:
                print(f' **REDIS: Error: no filter_val')
                return False

            res_db_list = [entry for entry in res_db_list if entry[kwargs["filter_key"]] == kwargs["filter_val"]]
            # print(f' **REDIS: filtered: {len(res_db_list)}')
        
        if kwargs["method"] == "get":
            return res_db_list
            
        if kwargs["method"] == "del_all":
            if len(res_db_list) > 0:
                update_i = 0
                for entry in [json.dumps(entry) for entry in res_db_list]:
                    r.lrem(kwargs["db_name"], 0, entry)
                    update_i = update_i + 1
                return res_db_list
            else:
                print(f' **REDIS: Error: no entry to delete for db_name: {kwargs["db_name"]}')
                return False
            
        if kwargs["method"] == "update":
            if len(res_db_list) > 0:
                update_i = 0
                for entry in [json.dumps(entry) for entry in res_db_list]:
                    r.lrem(kwargs["db_name"], 0, entry)
                    entry = json.loads(entry)
                    entry["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # entry["gpu"]["mem"] = f'blablabla + {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                    r.rpush(kwargs["db_name"], json.dumps(entry))
                    update_i = update_i + 1
                # print(f' **REDIS: updated ({update_i}/{len(res_db_list)})!')
                return res_db_list
            else:
                print(f' **REDIS: Error: no entry to update for db_name: {kwargs["db_name"]}')
                return False
                    
        if kwargs["method"] == "update2":
            print(f' ********************REDIS: "update2"')
            if not kwargs["update_val"]:
                print(f' ********************REDIS: ERROR NO "update_val"')
                return False
            else:
                print(f' ********************REDIS: SUCCES GOT "update_val": {kwargs["update_val"]}')
            
            if len(res_db_list) > 0:
                update_i = 0
                for entry in [json.dumps(entry) for entry in res_db_list]:
                    r.lrem(kwargs["db_name"], 0, entry)
                    entry = json.loads(entry)
                    # entry["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    entry["gpu"]["mem"] = f'mmhhmm {kwargs["update_val"]} |||| {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                    r.rpush(kwargs["db_name"], json.dumps(entry))
                    update_i = update_i + 1
                # print(f' **REDIS: updated ({update_i}/{len(res_db_list)})!')
                return res_db_list
            else:
                print(f' **REDIS: Error: no entry to update for db_name: {kwargs["db_name"]}')
                return False
        
        if kwargs["method"] == "save":
            if not kwargs["data"]:
                print(f' **REDIS: Error: no data to save')
                return False
            if not kwargs["data"]["uid"]:
                print(f' **REDIS: Error: no uid')
                return False
            else:
                print(f' **REDIS: YES GOT DATA UIDS!')

                
            # print(f' **REDIS: trying to get all uids ...')
            curr_uids = [entry["uid"] for entry in res_db_list]
            # print(f' **REDIS: found curr_uids: {len(curr_uids)}')

            if kwargs["data"]["uid"] in curr_uids:
                print(f' **REDIS: Error: vllm already saved!')
                return False

            
            save_data = kwargs["data"]
            
            
            # bbbbb
            data_obj = {
                "container_name": save_data.get("container_name", "err_container_name"),
                "uid": save_data.get("uid", "00000000000"),
                "State": {
                    "Status": "running"
                },
                "gpu": {
                    "mem": save_data.get("gpu", {}).get("mem", "err_gpu_mem")
                },
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }        
            r.rpush(kwargs["db_name"], json.dumps(data_obj))
            # print(f' **REDIS: saved!')
            return res_db_list
        
        return False
    
    except Exception as e:
        print(f' **REDIS: Error: {e}')
        return False










# test_call_save_vllm4 = {
#                 "db_name": REDIS_DB_VLLM,
#                 "method": "save",
#                 "select": "all",
#                 "data": res_vllms[3]
#             }

# test_call_save_vllm5 = {
#                 "db_name": REDIS_DB_VLLM,
#                 "method": "save",
#                 "select": "all",
#                 "data": res_vllms[4]
#             }

test_call_save = {
                "db_name": REDIS_DB_VLLM,
                "method": "save",
                "select": "all",
                "id": "3",
                "State": {
                    "Status": "running"
                },
                "ts": "0"
            }

test_call_get = {
                "db_name": REDIS_DB_VLLM,
                "method": "get",
                "select": "all"
            }

test_call_update = {
                "db_name": REDIS_DB_VLLM,
                "method": "update",
                "select": "all",
                "filter_key": "id",
                "filter_val": "3",
            }



test_call_update_gpu = {
                "db_name": REDIS_DB_VLLM,
                "method": "update",
                "select": "all",
                "filter_key": "id",
                "filter_val": "3",
            }




test_call_update_all = {
                "db_name": REDIS_DB_VLLM,
                "method": "update",
                "select": "all"
            }









test_call_save2= {
                "db_name": REDIS_DB_VLLM,
                "method": "save",
                "select": "all",
                "id": "container_vllm_oai",
                "State": {
                    "Status": "running"
                },
                "ts": "1"
            }



test_call_delete_all = {
                "db_name": REDIS_DB_VLLM,
                "method": "del_all",
                "select": "all"
            }

# aber muss net weil kannst .get("value"),0) bei return und nur 1 update geht auch wie b ei update fuilter eh
# status: starting, running, deploying, deleted ..

default_vllm = {
    "container_name": "vllm1",
    "uid": "123123",
    "created_hr": "123123",
    "created_ts": "123123",
    "created_by": "userasdf",
    "deleted_by": "userasdf",
    "used_by": ["userpublicasduzg","userasdf"],
    "expires": "created_ts+100000sec",
    "access": "public",
    
    "status": "running",

    "State": {
                "Status": "running"
            },
    "usage": {
                "tokens": ["111111","222222"],
                "prompts": ["NVIDIA RTX 3060","NVIDIA H100"],
                "prompts_response_time": [0.01,0.2,0.4,1.2],
                "prompts_response_time_per_token": [0.01,0.2,0.4,1.2],
                "gpu_util_per_sec_running": "gpu_util_per_sec_running",
                "mem_util_per_sec_running": "mem_util_per_sec_running"
            },
    "gpu": {
                "gpu_uuids": ["111111","222222"],
                "gpu_names": ["NVIDIA RTX 3060","NVIDIA H100"],
                "mem_util": "bissplitslash",
                "gpu_util": "bissplitslash",
                "temperature": "bissplitslash",
                "power_usage": "bissplitslash",
                "cuda_cores": "bissplitslash",
                "compute_capability": "bissplitslash",
                "status": "bissplitslash"
            },
    "model": {
                "id": "creater/name",
                "creater": "bissplitslash",
                "name": "nachsplitslash",
                "size": "nachsplitslash",
                "downloaded_hr": "bissplitslash",
                "downloaded_ts": "bissplitslash",
                "model_info": "bissplitslash",
                "model_configs": "bissplitslash"
            },
    "load_settings": {
                "load_component": "load_component"
            },
    "create_settings": {
                "create_component": "create_component"
            },
    "prompt_settings": {
                "prompt_component": "prompt_component"
            },
    "docker": {
                "container_id": "container_id",
                "container_name": "container_name",
                "container_status": "container_status",
                "container_info": "bissplitslash"
            },
    "ts": "0"
}

                # "name": gpu_info.get("name", "0"),
                # "mem_util": gpu_info.get("mem_util", "0"),
                # "timestamp": entry.get("timestamp", "0"),
                # "fan_speed": gpu_info.get("fan_speed", "0"),
                # "temperature": gpu_info.get("temperature", "0"),
                # "gpu_util": gpu_info.get("gpu_util", "0"),
                # "power_usage": gpu_info.get("power_usage", "0"),
                # "clock_info_graphics": gpu_info.get("clock_info_graphics", "0"),
                # "clock_info_mem": gpu_info.get("clock_info_mem", "0"),                
                # "cuda_cores": gpu_info.get("cuda_cores", "0"),
                # "compute_capability": gpu_info.get("compute_capability", "0"),
                # "current_uuid": gpu_info.get("current_uuid", "0"),
                # "gpu_i": entry.get("gpu_i", "0"),
                # "supported": gpu_info.get("supported", "0"),
                # "not_supported": gpu_info.get("not_supported", "0"),
                # "status": "ok"

                        # container_id = gr.Textbox(value=container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                        
                        # container_name = gr.Textbox(value=container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
            
                        # container_status = gr.Textbox(value=container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")



        
# print(f'__________________________________ get __________________________________')
# test_vllms = redis_connection(**test_call_get)
# print(f'________________________________ test_vllms: {test_vllms}')
# test_vllms_list_running = [c for c in test_vllms if c["State"]["Status"] == "running"]
# print(f'________________________________ test_vllms_list_running: {test_vllms_list_running}')
# print(f'________________________________________________________________________')

# test_vllms2 = get_vllms()
# print(f'________________________________ test_vllms2: {test_vllms2}')
# print(f'________________________________________________________________________')


print(f'__________________________________ delete ___________________________________')
redis_connection(**test_call_delete_all)
print(f'________________________________________________________________________')
print(f'')
time.sleep(0.05)

test_call_save_vllm1 = {
                "db_name": REDIS_DB_VLLM,
                "method": "save",
                "select": "all",
                "data": res_vllms[0]
            }

test_call_save_vllm2 = {
                "db_name": REDIS_DB_VLLM,
                "method": "save",
                "select": "all",
                "data": res_vllms[1]
            }

test_call_save_vllm3 = {
                "db_name": REDIS_DB_VLLM,
                "method": "save",
                "select": "all",
                "data": res_vllms[0]
            }
print(f'__________________________________ save ___________________________________')
print(f'______ calling mit res_vllms[0]: {res_vllms[0]}')
redis_connection(**test_call_save_vllm1)
print(f'______ calling mit res_vllms[1]: {res_vllms[1]}')
redis_connection(**test_call_save_vllm2)
print(f'______ calling mit res_vllms[0]: {res_vllms[0]}')
redis_connection(**test_call_save_vllm3)
print(f'________________________________________________________________________')
print(f'')
time.sleep(0.05)





# print(f'_________________________________ update ___________________________________')
# redis_connection(**test_call_update)
# print(f'________________________________________________________________________')
# print(f'')
# time.sleep(0.05)




def get_vllms_list():
    res_redis = redis_connection(**test_call_get)
    return res_redis

def update_vllms_list():
    res_redis = redis_connection(**test_call_update)
    return res_redis

    
def update_mem(new_mem):
    test_call_update2 = {
                "db_name": REDIS_DB_VLLM,
                "method": "update",
                "select": "all",
                "filter_key": "id",
                "filter_val": "3",
                "update_val": new_mem
                
            }
    res_redis = redis_connection(**test_call_update2)
    return res_redis

    
# def add_vllm4():
#     print(f'trying to uadd_vllm4 ...')
#     res_redis = redis_connection(**test_call_save_vllm4)
#     print(f'added add_vllm4! {res_redis}')
#     return res_redis

    
# def add_vllm5():
#     print(f'trying to uadd_vllm5 ...')
#     res_redis = redis_connection(**test_call_save_vllm5)
#     print(f'added add_vllm5! {res_redis}')
#     return res_redis











def selected_vllm_info(selected_radio):
    global res_vllms
    print(f'~~~~~~ [selected_vllm_info] REDIS_DB_VLLM: {REDIS_DB_VLLM}')
    print(f'~~~~~~ [selected_vllm_info] got selected_radio: {selected_radio}')

    req_vllm = {
        "db_name": REDIS_DB_VLLM,
        "method": "get",
        "select": "filter",
        "filter_key": "container_name",
        "filter_val": selected_radio,
    }
    print(f'~~~~~~ [selected_vllm_info] got selected_radio: {selected_radio}')
    
    
    res_vllm = redis_connection(**req_vllm)
    print(f'~~~~~~ [selected_vllm_info] got res_vllm: {res_vllm}')
    print(f'~~~~~~ [selected_vllm_info] got res_vllm[0]: {res_vllm[0]}')
    


        
    return f'{res_vllm}', f'{selected_radio}'




def toggle_test_vllms_create(vllm_list):
    # print(f'got vllm_list: {vllm_list}')
    if "Create New" in vllm_list:
        return (
            gr.Accordion(open=False,visible=False),
            gr.Button(visible=False),
            gr.Accordion(open=True,visible=True),
            gr.Button(visible=True)
        )

    return (
        gr.Accordion(open=True,visible=True),
        gr.Button(visible=True),    
        gr.Accordion(open=False,visible=False),
        gr.Button(visible=False)
    )




def get_vllm_list():
    # print(f'trying to get vllm ...')
    res_redis = redis_connection(**test_call_get)
    # choices_update = [c["container_name"] for c in res_redis]
    # print(f'got vllm! {res_redis}')
    return res_redis




def get_test_vllms():
    try:
        global test_vllms
        test_vllms = redis_connection(**test_call_get)
        # print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_vllms] GET! test_vllms: {test_vllms}')
        return test_vllms
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_vllms] {e}')
        return f'err {str(e)}'












def get_time():
    try:
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]'
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_vllms] {e}')
        return f'err {str(e)}'





# def get_test_vllms_radio(req_id):
#     try:
#         global test_vllms
        
#         test_call_get_filter = {
#                 "db_name": req_db,
#                 "method": "get",
#                 "select": "filter",
#                 "filter_key": "id",
#                 "filter_val": req_id,
#         }
        
        
        
#         test_vllms = redis_connection(**test_call_get_filter)
#         print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_test_vllms_radio] req_id: {req_id}')
#         return test_vllms
    
#     except Exception as e:
#         print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_vllms] {e}')
#         return f'err {str(e)}'






def refresh_container():
    try:
        global docker_container_list
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/docker', json={"method": "list"})
        docker_container_list = response.json()
        return docker_container_list
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'err {str(e)}'

            
@dataclass
class VllmCreateComponents:
    method: gr.Textbox
    image: gr.Textbox
    runtime: gr.Textbox
    shm_size: gr.Slider
    port: gr.Slider
    max_model_len: gr.Slider
    tensor_parallel_size: gr.Number
    gpu_memory_utilization: gr.Slider
    
    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]

@dataclass
class VllmCreateValues:
    method: str
    image: str
    runtime: str
    shm_size: int
    port: int
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: int



@dataclass
class VllmLoadComponents:
    method: gr.Textbox
    vllmcontainer: gr.Textbox
    port: gr.Slider
    image: gr.Textbox
    max_model_len: gr.Slider
    tensor_parallel_size: gr.Number
    gpu_memory_utilization: gr.Slider
    
    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]

@dataclass
class VllmLoadValues:
    method: str
    vllmcontainer: str
    port: int
    image: str
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: int



@dataclass
class PromptComponents:
    vllmcontainer: gr.Radio
    port: gr.Slider
    prompt: gr.Textbox
    top_p: gr.Slider
    temperature: gr.Slider
    max_tokens: gr.Slider
    
    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]

@dataclass
class PromptValues:
    vllmcontainer: str
    port: int
    prompt: str
    top_p: int
    temperature: int
    max_tokens: int


BACKEND_URL = f'http://container_backend:{os.getenv("BACKEND_PORT")}/docker'
AUDIO_URL = f'http://container_audio:{os.getenv("AUDIO_PORT")}/a'
# AUDIO_URL = f'http://container_audio:{os.getenv("AUDIO_PORT")}/t'


def toggle_vllm_load_create(vllm_list):
    
    if "Create New" in vllm_list:
        return (
            gr.Accordion(open=False,visible=False),
            gr.Button(visible=False),
            gr.Accordion(open=True,visible=True),
            gr.Button(visible=True)
        )

    return (
        gr.Accordion(open=True,visible=True),
        gr.Button(visible=True),    
        gr.Accordion(open=False,visible=False),
        gr.Button(visible=False)
    )

def toggle_vllm_prompt(vllm_list_prompt):
    global PROMPT
    PROMPT = 'asdasdasdasdasdasdasd'
    return gr.Textbox(value="9999")   
    # if "Create New" in vllm_list_prompt:
    #     return (
    #         gr.Accordion(open=False,visible=False),
    #         gr.Button(visible=False),
    #         gr.Accordion(open=True,visible=True),
    #         gr.Button(visible=True)
    #     )

    # return (
    #     gr.Accordion(open=True,visible=True),
    #     gr.Button(visible=True),    
    #     gr.Accordion(open=False,visible=False),
    #     gr.Button(visible=False)
    # )

   
    
def llm_load(*params):
    
    try:
        global SELECTED_MODEL_ID
        print(f' >>> llm_load SELECTED_MODEL_ID: {SELECTED_MODEL_ID} ')
        print(f' >>> llm_load got params: {params} ')
        logging.exception(f'[llm_load] >> SELECTED_MODEL_ID: {SELECTED_MODEL_ID} ')
        logging.exception(f'[llm_load] >> got params: {params} ')
                
        req_params = VllmLoadComponents(*params)

        response = requests.post(BACKEND_URL, json={
            "method":req_params.method,
            "vllmcontainer":req_params.vllmcontainer,
            "image":req_params.image,
            "port":req_params.port,
            "model":SELECTED_MODEL_ID,
            "tensor_parallel_size":req_params.tensor_parallel_size,
            "gpu_memory_utilization":req_params.gpu_memory_utilization,
            "max_model_len":req_params.max_model_len
        }, timeout=REQUEST_TIMEOUT)


        if response.status_code == 200:
            print(f' [llm_load] >> got response == 200 building json ... {response} ')
            logging.exception(f'[llm_load] >> got response == 200 building json ...  {response} ')
            res_json = response.json()        
            print(f' [llm_load] >> GOT RES_JSON: SELECTED_MODEL_ID: {res_json} ')         
            return f'{res_json}'
        else:
            logging.exception(f'[llm_load] Request Error: {response}')
            return f'Request Error: {response}'
    
    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'{e}'
        
    
def llm_create(*params):
    
    try:
        global SELECTED_MODEL_ID
        print(f' >>> llm_create SELECTED_MODEL_ID: {SELECTED_MODEL_ID} ')
        print(f' >>> llm_create got params: {params} ')
        logging.exception(f'[llm_create] >> SELECTED_MODEL_ID: {SELECTED_MODEL_ID} ')
        logging.exception(f'[llm_create] >> got params: {params} ')
                
        req_params = VllmCreateComponents(*params)

        response = requests.post(BACKEND_URL, json={
            "method":req_params.method,
            "image":req_params.image,
            "runtime":req_params.runtime,
            "shm_size":f'{str(req_params.shm_size)}gb',
            "port":req_params.port,
            "model":SELECTED_MODEL_ID,
            "tensor_parallel_size":req_params.tensor_parallel_size,
            "gpu_memory_utilization":req_params.gpu_memory_utilization,
            "max_model_len":req_params.max_model_len
        }, timeout=REQUEST_TIMEOUT)


        if response.status_code == 200:
            print(f' [llm_create] >> got response == 200 building json ... {response} ')
            logging.exception(f'[llm_create] >> got response == 200 building json ...  {response} ')
            res_json = response.json()        
            print(f' [llm_create] >> GOT RES_JSON: SELECTED_MODEL_ID: {res_json} ')         
            return f'{res_json}'
        else:
            logging.exception(f'[llm_create] Request Error: {response}')
            return f'Request Error: {response}'
    
    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'{e}'
    
        
def llm_prompt(*params):
    
    try:
        global SELECTED_MODEL_ID
        print(f' >>> llm_prompt SELECTED_MODEL_ID: {SELECTED_MODEL_ID} ')
        print(f' >>> llm_prompt got params: {params} ')
        logging.info(f'[llm_prompt] >> SELECTED_MODEL_ID: {SELECTED_MODEL_ID} ')
        logging.info(f'[llm_prompt] >> got params: {params} ')

        req_params = PromptComponents(*params)

        DEFAULTS_PROMPT = {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "vllmcontainer": "container_vllm_xoo",
            "port": 1370,
            "prompt": "Tell a joke",
            "top_p": 0.95,
            "temperature": 0.8,
            "max_tokens": 150
        }

        response = requests.post(BACKEND_URL, json={
            "method":"generate",
            "model":SELECTED_MODEL_ID,
            "vllmcontainer":getattr(req_params, "vllmcontainer", DEFAULTS_PROMPT["vllmcontainer"]),
            "port":getattr(req_params, "port", DEFAULTS_PROMPT["port"]),
            "prompt": getattr(req_params, "prompt", DEFAULTS_PROMPT["prompt"]),
            "top_p":getattr(req_params, "top_p", DEFAULTS_PROMPT["top_p"]),
            "temperature":getattr(req_params, "temperature", DEFAULTS_PROMPT["temperature"]),
            "max_tokens":getattr(req_params, "max_tokens", DEFAULTS_PROMPT["max_tokens"])
        }, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            print(f' !?!?!?!? [llm_prompt] got response == 200 building json ... {response} ')
            logging.info(f'!?!?!?!? [llm_prompt] got response == 200 building json ...  {response} ')
            res_json = response.json()        
            print(f' !?!?!?!? [llm_prompt] GOT RES_JSON: llm_prompt SELECTED_MODEL_ID: {res_json} ')
            logging.info(f'!?!?!?!? [llm_prompt] GOT RES_JSON: {res_json} ')
            if res_json["result_status"] != 200:
                print(f' !?!?!?!? [llm_prompt] res_json["result_status"] != 200: {res_json} ')
                logging.exception(f'[llm_prompt] Response Error: {res_json["result_data"]}')
                return f'{res_json}'
            return f'{res_json["result_data"]}'
        else:
            logging.exception(f'[llm_prompt] Request Error: {response}')
            return f'Request Error: {response}'
    
    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'{e}'
    



def download_from_hf_hub(selected_model_id):
    try:
        selected_model_id_arr = str(selected_model_id).split('/')
        print(f'selected_model_id_arr {selected_model_id_arr}...')       
        model_path = snapshot_download(
            repo_id=selected_model_id,
            local_dir=f'/models/{selected_model_id_arr[0]}/{selected_model_id_arr[1]}'
        )
        return f'Saved to {model_path}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'download error: {e}'


download_info_prev_bytes_recv = 0   
download_info_current_model_bytes_recv = 0    
 
def download_info(req_model_size, progress=gr.Progress()):
    global download_info_prev_bytes_recv
    global download_info_current_model_bytes_recv
    download_info_prev_bytes_recv = 0
    download_info_current_model_bytes_recv = 0
    progress(0, desc="Initializing ...")
    progress(0.01, desc="Calculating Download Time ...")
    
    avg_dl_speed_val = 0
    avg_dl_speed = []
    for i in range(0,5):
        net_io = psutil.net_io_counters()
        bytes_recv = net_io.bytes_recv
        download_speed = bytes_recv - download_info_prev_bytes_recv
        download_speed_mbit_s = (download_speed * 8) / (1024 ** 2) 
        
        download_info_prev_bytes_recv = int(bytes_recv)
        download_info_current_model_bytes_recv = download_info_current_model_bytes_recv + download_info_prev_bytes_recv
        avg_dl_speed.append(download_speed)
        avg_dl_speed_val = sum(avg_dl_speed)/len(avg_dl_speed)
        logging.info(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')
        print(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')  
        time.sleep(1)
    
    logging.info(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')
    print(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')  



    calc_mean = lambda data: np.mean([x for x in data if (np.percentile(data, 25) - 1.5 * (np.percentile(data, 75) - np.percentile(data, 25))) <= x <= (np.percentile(data, 75) + 1.5 * (np.percentile(data, 75) - np.percentile(data, 25)))]) if data else 0


    avg_dl_speed_val = calc_mean(avg_dl_speed)
        
    
    logging.info(f' **************** [download_info] avg_dl_speed_val: {avg_dl_speed_val}')
    print(f' **************** [download_info] avg_dl_speed_val: {avg_dl_speed_val}')    

    est_download_time_sec = int(req_model_size)/int(avg_dl_speed_val)
    logging.info(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')
    print(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')

    est_download_time_sec = int(est_download_time_sec)
    logging.info(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')
    print(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')

    logging.info(f' **************** [download_info] zzz waiting for download_complete_event zzz waiting {est_download_time_sec}')
    print(f' **************** [download_info] zzz waiting for download_complete_event zzz waiting {est_download_time_sec}')
    current_dl_arr = []
    for i in range(0,est_download_time_sec):
        if len(current_dl_arr) > 5:
            current_dl_arr = []
        net_io = psutil.net_io_counters()
        bytes_recv = net_io.bytes_recv
        download_speed = bytes_recv - download_info_prev_bytes_recv
        current_dl_arr.append(download_speed)
        print(f' &&&&&&&&&&&&&& current_dl_arr: {current_dl_arr}')
        if all(value < 10000 for value in current_dl_arr[-4:]):
            print(f' &&&&&&&&&&&&&& DOWNLOAD FINISH EHH??: {current_dl_arr}')
            yield f'Progress: 100%\nFiniiiiiiiish!'
            return
            
        download_speed_mbit_s = (download_speed * 8) / (1024 ** 2)
        
        download_info_prev_bytes_recv = int(bytes_recv)
        download_info_current_model_bytes_recv = download_info_current_model_bytes_recv + download_info_prev_bytes_recv

        progress_percent = (i + 1) / est_download_time_sec
        progress(progress_percent, desc=f"Downloading ... {download_speed_mbit_s:.2f} MBit/s")

        time.sleep(1)
    logging.info(f' **************** [download_info] LOOP DONE!')
    print(f' **************** [download_info] LOOP DONE!')
    yield f'Progress: 100%\nFiniiiiiiiish!'


def parallel_download(selected_model_size, model_dropdown):
    # Create threads for both functions
    thread_info = threading.Thread(target=download_info, args=(selected_model_size,))
    thread_hub = threading.Thread(target=download_from_hf_hub, args=(model_dropdown,))

    # Start both threads
    thread_info.start()
    thread_hub.start()

    # Wait for both threads to finish
    thread_info.join()
    thread_hub.join()

    return "Download finished!"




def change_tab(n):
    return gr.Tabs(selected=n)

def toggle_test_vllms(req_id):
    try:
        global test_vllms
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_test_vllms_radio] req_id: {req_id}')
        
        test_call_get_filter2 = {
            "db_name": "asd",
            "method": "get",
            "select": "filter",
            "filter_key": "id",
            "filter_val": req_id,
        }
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        
        test_vllms = redis_connection(**test_call_get_filter2)
        
        # Add proper error handling for the redis response
        if test_vllms is False:
            return "Redis operation failed"

            
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_test_vllms_radio] test_vllms: {test_vllms}')

        
        print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_test_vllms_radio] test_vllms[0]: {test_vllms[0]}')
        print("ccccccccccccccccccccccccccccccccccccccc")
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_test_vllms_radio] test_vllms[0]["id"]: {test_vllms[0]["id"]}')
        print("ddddddddddddddddddddd")
        
        return test_vllms[0]["id"],f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_vllms] {e}')
        return f'err {str(e)}'




def create_app():
    with gr.Blocks() as app:
        
        
        vllm_state = gr.State([])
        container_state = gr.State(value=[])


        def get_vllm():
            try:
                vllm = get_vllm_list() # 1x docker 1x vllm
                return vllm
            except Exception as e:
                print(f'[get_vllm] Error {e}')
                return []
                
        def get_container():
            try:
                # docker_container = get_docker_container_list() # 1x docker 1x vllm
                docker_container = docker_api("list",None)
                return docker_container
            except Exception as e:
                print(f'[get_container] Error {e}')
                return []
        
        app.load(get_vllm, outputs=[vllm_state])
        app.load(get_container, outputs=[container_state])
        
        txt_lambda_log_helper = gr.Textbox(value="logs")
        txt_lambda_start_helper = gr.Textbox(value="start")
        txt_lambda_stop_helper = gr.Textbox(value="stop")
        txt_lambda_delete_helper = gr.Textbox(value="delete")
        
        @gr.render(inputs=[container_state])
        def render_container(container_list):
            if container_list:
                docker_container_list_sys_running = [c for c in container_list if c["State"]["Status"] == "running" and c["Name"] in [f'/container_redis',f'/container_backend', f'/container_frontend', f'/container_audio']]
                docker_container_list_sys_not_running = [c for c in docker_container_list if c["State"]["Status"] != "running" and c["Name"] in [f'/container_redis',f'/container_backend', f'/container_frontend', f'/container_audio']]
                
                docker_container_list_vllm_running = [c for c in docker_container_list if c["State"]["Status"] == "running" and c["Name"] not in [f'/container_redis',f'/container_backend', f'/container_frontend', f'/container_audio']]
                docker_container_list_vllm_not_running = [c for c in docker_container_list if c["State"]["Status"] != "running" and c["Name"] not in [f'/container_redis',f'/container_backend', f'/container_frontend', f'/container_audio']]
                
                with gr.Accordion(("System Container"), open=False, visible=True) as acc_prompt:
                    with gr.Tabs() as tabs:
                        for container in docker_container_list_sys_running:
                            print(f'HHHHHHHHUM container["Name"]: {container["Name"]}')
                            if container["Name"] == "/container_redis":
                                print(f'HHHHHHHHUM YEYE REDIS')
                                # Create unique ID for each tab
                                tab_id = f"tab_{container["Id"][:12]}"
                                with gr.TabItem(container["Name"][1:], id=tab_id):
                                    with gr.Row():
                                        container_id = gr.Textbox(value=container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                                        
                                        container_name = gr.Textbox(value=container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")
                                        
                                        container_status = gr.Textbox(value=container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                            
                                        container_ports = gr.Textbox(value=next(iter(container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                                    with gr.Row():
                                        container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)
                                        
                                    with gr.Row():                                      
                                        btn_logs_docker_open = gr.Button("Docker Log", scale=0)
                                        btn_logs_docker_close = gr.Button("Close Docker Log", scale=0, visible=False)     
                                        
                                        btn_logs_docker_open.click(
                                            docker_api,
                                            [txt_lambda_log_helper,container_id],
                                            [container_log_out]
                                        ).then(
                                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                                        )
                                        
                                        btn_logs_docker_close.click(
                                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                                        )
                                        

                                        
                                        start_btn = gr.Button("Start", scale=0)
                                        stop_btn = gr.Button("Stop", scale=0)
                                        delete_btn = gr.Button("Delete", scale=0, variant="stop")

                                        start_btn.click(
                                            docker_api,
                                            [txt_lambda_start_helper,container_id],
                                            [container_state]
                                        ).then(
                                            refresh_container,
                                            [container_state]
                                        )
                                        
                                        stop_btn.click(
                                            docker_api,
                                            [txt_lambda_stop_helper,container_id],
                                            [container_state]
                                        ).then(
                                            refresh_container,
                                            outputs=[container_state]
                                        )

                                        delete_btn.click(
                                            docker_api,
                                            [txt_lambda_delete_helper,container_id],
                                            [container_state]
                                        ).then(
                                            refresh_container,
                                            outputs=[container_state]
                                        )
                            else:
                                # Create unique ID for each tab
                                tab_id = f"tab_{container["Id"][:12]}"
                                with gr.TabItem(container["Name"][1:], id=tab_id):
                                    with gr.Row():
                                        container_id = gr.Textbox(value=container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                                        
                                        container_name = gr.Textbox(value=container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")
                                        
                                        container_status = gr.Textbox(value=container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                            
                                        container_ports = gr.Textbox(value=next(iter(container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                                    with gr.Row():
                                        container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)
                                        
                                    with gr.Row():                                    
                                        btn_logs_file_open = gr.Button("Log File", scale=0)
                                        btn_logs_file_close = gr.Button("Close Log File", scale=0, visible=False)   
                                        btn_logs_docker_open = gr.Button("Docker Log", scale=0)
                                        btn_logs_docker_close = gr.Button("Close Docker Log", scale=0, visible=False)     

                                        btn_logs_file_open.click(
                                            load_log_file,
                                            [container_name],
                                            [container_log_out]
                                        ).then(
                                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                                        )
                                        
                                        btn_logs_file_close.click(
                                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                                        )
                                        
                                        btn_logs_docker_open.click(
                                            docker_api,
                                            [txt_lambda_log_helper,container_id],
                                            [container_log_out]
                                        ).then(
                                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                                        )
                                        
                                        btn_logs_docker_close.click(
                                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                                        )
                                        

                                        
                                        start_btn = gr.Button("Start", scale=0)
                                        stop_btn = gr.Button("Stop", scale=0)
                                        delete_btn = gr.Button("Delete", scale=0, variant="stop")

                                        start_btn.click(
                                            docker_api,
                                            [txt_lambda_start_helper,container_id],
                                            [container_state]
                                        ).then(
                                            refresh_container,
                                            [container_state]
                                        )
                                        
                                        stop_btn.click(
                                            docker_api,
                                            [txt_lambda_stop_helper,container_id],
                                            [container_state]
                                        ).then(
                                            refresh_container,
                                            outputs=[container_state]
                                        )

                                        delete_btn.click(
                                            docker_api,
                                            [txt_lambda_delete_helper,container_id],
                                            [container_state]
                                        ).then(
                                            refresh_container,
                                            outputs=[container_state]
                                        )

                with gr.Accordion(("vLLM Container"), open=False, visible=True) as acc_prompt:
                    with gr.Tabs() as tabs3:
                        for container in docker_container_list_vllm_running:
                            # Create unique ID for each tab
                            tab_id = f"tab_{container["Id"][:12]}"
                            with gr.TabItem(container["Name"][1:], id=tab_id):
                                with gr.Row():
                                    container_id = gr.Textbox(value=container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                                    
                                    container_name = gr.Textbox(value=container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")
                                    
                                    container_status = gr.Textbox(value=container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                        
                                    container_ports = gr.Textbox(value=next(iter(container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                    
                                    
                                with gr.Row():
                                    container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)

                                with gr.Row():
                                    btn_logs_docker_open = gr.Button("Docker Log", scale=0)
                                    btn_logs_docker_close = gr.Button("Close Docker Log", scale=0, visible=False)     
                                    
                                    btn_logs_docker_open.click(
                                        docker_api,
                                        [txt_lambda_log_helper,container_id],
                                        [container_log_out]
                                    ).then(
                                        lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                                    )
                                    
                                    btn_logs_docker_close.click(
                                        lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                                    )
                                    
                                    start_btn = gr.Button("Start", scale=0)
                                    stop_btn = gr.Button("Stop", scale=0)
                                    delete_btn = gr.Button("Delete", scale=0, variant="stop")

                                    start_btn.click(
                                        docker_api,
                                        [txt_lambda_start_helper,container_id],
                                        [container_state]
                                    ).then(
                                        refresh_container,
                                        [container_state]
                                    )                                    
                                    
                                    stop_btn.click(
                                        docker_api,
                                        [txt_lambda_stop_helper,container_id],
                                        [container_state]
                                    ).then(
                                        refresh_container,
                                        outputs=[container_state]
                                    )
                    
                                    delete_btn.click(
                                        docker_api,
                                        [txt_lambda_delete_helper,container_id],
                                        [container_state]
                                    ).then(
                                        refresh_container,
                                        outputs=[container_state]
                                    )
                        
            
            else:
                gr.Markdown("No containers available")
        
        refresh_btn = gr.Button("Refresh Containers")
        refresh_btn.click(
            refresh_container,
            outputs=[container_state]
        ) 

        # bbbbb
        radio_state = gr.State("")
          
        @gr.render(inputs=[vllm_state,radio_state])
        def render_vllm(vllm_list,radio_state_val):
            

            
            print(f'radio_state_val: {radio_state_val}')

            with gr.Row():
                for current_vllm in vllm_list:
                    with gr.Row():
                        print(f'current_vllm')
                        print(current_vllm)
                        vllm_selected = gr.Radio([f'{current_vllm["container_name"]}'], value=radio_state_val, interactive=True, label=f'{current_vllm["uid"]} {current_vllm["ts"]} | {current_vllm["gpu"]["mem"]} | {current_vllm["gpu"]["mem"]} ',info=f'{current_vllm["State"]["Status"]} {current_vllm["ts"]}')

                                        
                        # container_name = gr.Textbox(value=current_vllm["container_name"], interactive=False, elem_classes="table-cell", label="Container Name", info="Container Name")  

                                        
                        # ts = gr.Textbox(value=current_vllm["ts"], interactive=False, elem_classes="table-cell", label="ts", show_label=False)  
                        

            
                        vllm_selected.change(
                            selected_vllm_info,
                            [vllm_selected],
                            [selected_vllm_uuid, radio_state]
                        )
                        
        
        with gr.Accordion(("Selected vLLM Additional Information"), open=True, visible=True) as acc_prompt:
                                


            selected_vllm_uuid = gr.Textbox(label="selected_vllm_uuid",value=f'nix bla')


            with gr.Accordion(("Model Parameters"), open=False):
                    with gr.Row():
                        selected_model_id = gr.Textbox(label="id")
                        selected_model_container_name = gr.Textbox(label="container_name")
                        
                        
                    with gr.Row():
                        selected_model_architectures = gr.Textbox(label="architectures")
                        selected_model_pipeline_tag = gr.Textbox(label="pipeline_tag")
                        selected_model_transformers = gr.Textbox(label="transformers")
                        
                        
                    with gr.Row():
                        selected_model_model_type = gr.Textbox(label="model_type")
                        selected_model_quantization = gr.Textbox(label="quantization")
                        selected_model_torch_dtype = gr.Textbox(label="torch_dtype")
                        selected_model_size = gr.Textbox(label="size")
                        selected_model_hidden_size = gr.Textbox(label="hidden_size", visible=False)

                    with gr.Row():
                        selected_model_private = gr.Textbox(label="private")
                        selected_model_gated = gr.Textbox(label="gated")
                        selected_model_downloads = gr.Textbox(label="downloads")
                                          
                        
                        
                    
                    with gr.Accordion(("Model Configs"), open=False):
                        with gr.Row():
                            selected_model_search_data = gr.Textbox(label="search_data", lines=20, elem_classes="table-cell")
                        with gr.Row():
                            selected_model_hf_data = gr.Textbox(label="hf_data", lines=20, elem_classes="table-cell")
                        with gr.Row():
                            selected_model_config_data = gr.Textbox(label="config_data", lines=20, elem_classes="table-cell")
                        
        


            with gr.Accordion(("Prompt Parameters"), open=False, visible=True) as acc_prompt:
                                    
                global PROMPT
                vllmcontainer=gr.Radio(["container_vllm_xoo", "container_vllm_oai", "Create New"], value="container_vllm_oai", show_label=False, info="Select a vllms_prompt or create a new one. Where?"),
                port=gr.Slider(1370, 1380, step=1, value=1371, label="port", info=f"Choose a port."),
                prompt = gr.Textbox(placeholder=f'{PROMPT}', value=f'{PROMPT}', label="Prompt", show_label=True, visible=True),
                top_p=gr.Slider(0.01, 1.0, step=0.01, value=0.95, label="top_p", info=f'Float that controls the cumulative probability of the top tokens to consider'),
                temperature=gr.Slider(0.0, 0.99, step=0.01, value=0.8, label="temperature", info=f'Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling'),
                max_tokens=gr.Slider(50, 2500, step=25, value=150, label="max_tokens", info=f'Maximum number of tokens to generate per output sequence')

            with gr.Accordion(("Load vLLM Parameters"), open=False, visible=True) as acc_load:
                    method=gr.Textbox(value="load", label="method", info=f"yee the req_method."),
                    vllmcontainer=gr.Textbox(value="container_vllm_xoo", label="vllmcontainer", info=f"Select a container name which is running vLLM"),
                    # vllmcontainer=gr.Radio(["container_vllm_xoo", "container_vllm_oai", "Create New"], value="container_vllm_xoo", show_label=False, info="Select a vllms_prompt or create a new one. Where?"),
                    port=gr.Slider(1370, 1380, step=1, value=1370, label="port", info=f"Choose a port."),
                    image=gr.Textbox(value="xoo4foo/zzvllm46:latest", label="image", info=f"Dockerhub vLLM image"),
                                                                    
                    max_model_len=gr.Slider(1024, 8192, step=1024, value=1024, label="max_model_len", info=f"Model context length. If unspecified, will be automatically derived from the model config."),
                    tensor_parallel_size=gr.Number(1, 8, value=1, label="tensor_parallel_size", info=f"Number of tensor parallel replicas."),
                    gpu_memory_utilization=gr.Slider(0.2, 0.99, value=0.87, label="gpu_memory_utilization", info=f"The fraction of GPU memory to be used for the model executor, which can range from 0 to 1.")
    

            
            
            with gr.Accordion(("Create vLLM Parameters"), open=False, visible=True) as acc_create:
                method=gr.Textbox(value="create", label="method", info=f"yee the req_method."),
                
                image=gr.Textbox(value="xoo4foo/zzvllm46:latest", label="image", info=f"Dockerhub vLLM image"),
                runtime=gr.Textbox(value="nvidia", label="runtime", info=f"Container runtime"),
                shm_size=gr.Slider(1, 320, step=1, value=8, label="shm_size", info=f'Maximal GPU Memory in GB'),
                
                port=gr.Slider(1370, 1380, step=1, value=1370, label="port", info=f"Choose a port."),                        
                
                max_model_len=gr.Slider(1024, 8192, step=1024, value=1024, label="max_model_len", info=f"Model context length. If unspecified, will be automatically derived from the model config."),
                tensor_parallel_size=gr.Number(1, 8, value=1, label="tensor_parallel_size", info=f"Number of tensor parallel replicas."),
                gpu_memory_utilization=gr.Slider(0.2, 0.99, value=0.87, label="gpu_memory_utilization", info=f"The fraction of GPU memory to be used for the model executor, which can range from 0 to 1.")

    
        
        # btn_add_vllm4 = gr.Button("add vllm4")
        # btn_add_vllm4.click(
        #     add_vllm4,
        #     None,
        #     [container_state]
        # )
        
        # btn_add_vllm5 = gr.Button("add vllm5")
        # btn_add_vllm5.click(
        #     add_vllm5,
        #     None,
        #     [container_state]
        # )
        
        
        vllm_radio_timer = gr.Timer(5,active=True)
        vllm_radio_timer.tick(
            update_vllms_list,
            None,
            [vllm_state],
            show_progress=False
        )
        
        





        gr.Markdown(
        """
        # Welcome!
        Select a Hugging Face model and deploy it on a port
        # Hallo!
        Testen Sie LLM AI Models auf verschiedenen Ports mit custom vLLM images
        **Note**: _[vLLM supported models list](https://docs.vllm.ai/en/latest/models/supported_models.html)_        
        """)
        input_search = gr.Textbox(placeholder="Enter Hugging Face model name or tag", label=f'found 0 models', show_label=False, autofocus=True)
        btn_search = gr.Button("Search")
        btn_tested_models = gr.Button("Load tested models")
        










        with gr.Row(visible=False) as row_model_select:
            model_dropdown = gr.Dropdown(choices=[''], interactive=True, show_label=False)
        with gr.Row(visible=False) as row_model_info:
            with gr.Column(scale=4):
                with gr.Accordion(("Model Parameters"), open=False):
                    with gr.Row():
                        selected_model_id = gr.Textbox(label="id")
                        selected_model_container_name = gr.Textbox(label="container_name")
                        
                        
                    with gr.Row():
                        selected_model_architectures = gr.Textbox(label="architectures")
                        selected_model_pipeline_tag = gr.Textbox(label="pipeline_tag")
                        selected_model_transformers = gr.Textbox(label="transformers")
                        
                        
                    with gr.Row():
                        selected_model_model_type = gr.Textbox(label="model_type")
                        selected_model_quantization = gr.Textbox(label="quantization")
                        selected_model_torch_dtype = gr.Textbox(label="torch_dtype")
                        selected_model_size = gr.Textbox(label="size")
                        selected_model_hidden_size = gr.Textbox(label="hidden_size", visible=False)

                    with gr.Row():
                        selected_model_private = gr.Textbox(label="private")
                        selected_model_gated = gr.Textbox(label="gated")
                        selected_model_downloads = gr.Textbox(label="downloads")
                                          
                        
                        
                    
                    with gr.Accordion(("Model Configs"), open=False):
                        with gr.Row():
                            selected_model_search_data = gr.Textbox(label="search_data", lines=20, elem_classes="table-cell")
                        with gr.Row():
                            selected_model_hf_data = gr.Textbox(label="hf_data", lines=20, elem_classes="table-cell")
                        with gr.Row():
                            selected_model_config_data = gr.Textbox(label="config_data", lines=20, elem_classes="table-cell")
                        
        
        
        
        vllms=gr.Radio(["vLLM1", "vLLM2", "Create New"], value="vLLM1", show_label=False, info="Select a vLLM or create a new one. Where?")
        current_tab = gr.Number(value=0, visible=False)
        with gr.Tabs() as tabs:
            with gr.TabItem("Download", id=0):
                with gr.Row(visible=True) as row_vllm_download:
                    with gr.Column(scale=4):
                        download_status = gr.Textbox(show_label=False, visible=False)
                
                    with gr.Column(scale=1):                
                        with gr.Row(visible=True) as row_download:
                            btn_dl = gr.Button("DOWNLOAD", variant="primary")

            with gr.TabItem("Load", id=1):
                with gr.Row(visible=True) as row_vllm_load:
                    with gr.Column(scale=4):
                        with gr.Accordion(("Load vLLM Parameters"), open=True, visible=True) as acc_load:
                            vllm_load_components = VllmLoadComponents(

                                method=gr.Textbox(value="load", label="method", info=f"yee the req_method."),
                                vllmcontainer=gr.Textbox(value="container_vllm_xoo", label="vllmcontainer", info=f"Select a container name which is running vLLM"),
                                # vllmcontainer=gr.Radio(["container_vllm_xoo", "container_vllm_oai", "Create New"], value="container_vllm_xoo", show_label=False, info="Select a vllms_prompt or create a new one. Where?"),
                                port=gr.Slider(1370, 1380, step=1, value=1370, label="port", info=f"Choose a port."),
                                image=gr.Textbox(value="xoo4foo/zzvllm46:latest", label="image", info=f"Dockerhub vLLM image"),
                                                                                
                                max_model_len=gr.Slider(1024, 8192, step=1024, value=1024, label="max_model_len", info=f"Model context length. If unspecified, will be automatically derived from the model config."),
                                tensor_parallel_size=gr.Number(1, 8, value=1, label="tensor_parallel_size", info=f"Number of tensor parallel replicas."),
                                gpu_memory_utilization=gr.Slider(0.2, 0.99, value=0.87, label="gpu_memory_utilization", info=f"The fraction of GPU memory to be used for the model executor, which can range from 0 to 1.")
                            )
                
                    with gr.Column(scale=1):
                        with gr.Row(visible=True) as vllm_load_actions:
                            btn_load = gr.Button("DEPLOY")

            with gr.TabItem("Create", id=2):
                with gr.Row(visible=True) as row_vllm_create:
                    with gr.Column(scale=4):                         
                        with gr.Accordion(("Create vLLM Parameters"), open=True, visible=True) as acc_create:
                            vllm_create_components = VllmCreateComponents(

                                method=gr.Textbox(value="create", label="method", info=f"yee the req_method."),
                                
                                image=gr.Textbox(value="xoo4foo/zzvllm46:latest", label="image", info=f"Dockerhub vLLM image"),
                                runtime=gr.Textbox(value="nvidia", label="runtime", info=f"Container runtime"),
                                shm_size=gr.Slider(1, 320, step=1, value=8, label="shm_size", info=f'Maximal GPU Memory in GB'),
                                
                                port=gr.Slider(1370, 1380, step=1, value=1370, label="port", info=f"Choose a port."),                        
                                
                                max_model_len=gr.Slider(1024, 8192, step=1024, value=1024, label="max_model_len", info=f"Model context length. If unspecified, will be automatically derived from the model config."),
                                tensor_parallel_size=gr.Number(1, 8, value=1, label="tensor_parallel_size", info=f"Number of tensor parallel replicas."),
                                gpu_memory_utilization=gr.Slider(0.2, 0.99, value=0.87, label="gpu_memory_utilization", info=f"The fraction of GPU memory to be used for the model executor, which can range from 0 to 1.")
                            )
                
                    with gr.Column(scale=1):
                        with gr.Row(visible=True) as vllm_create_actions:
                            btn_create = gr.Button("CREATE", variant="primary")
                            btn_create_close = gr.Button("CANCEL")


            with gr.TabItem("Prompt", id=3):
                with gr.Row(visible=True) as row_vllm_prompt:
                    with gr.Column(scale=2):
                        with gr.Accordion(("Prompt Parameters"), open=True, visible=True) as acc_prompt:

                            llm_prompt_components = PromptComponents(
                                vllmcontainer=gr.Radio(["container_vllm_xoo", "container_vllm_oai", "Create New"], value="container_vllm_oai", show_label=False, info="Select a vllms_prompt or create a new one. Where?"),
                                port=gr.Slider(1370, 1380, step=1, value=1371, label="port", info=f"Choose a port."),
                                prompt = gr.Textbox(placeholder=f'{PROMPT}', value=f'{PROMPT}', label="Prompt", show_label=True, visible=True),
                                top_p=gr.Slider(0.01, 1.0, step=0.01, value=0.95, label="top_p", info=f'Float that controls the cumulative probability of the top tokens to consider'),
                                temperature=gr.Slider(0.0, 0.99, step=0.01, value=0.8, label="temperature", info=f'Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling'),
                                max_tokens=gr.Slider(50, 2500, step=25, value=150, label="max_tokens", info=f'Maximum number of tokens to generate per output sequence')
                            )
                            testtext = gr.Textbox(placeholder="testplaceholder", value="testvalue", show_label=False, autofocus=True)
                            with gr.Row():
                                vllmsprompt=gr.Radio(["vLLM1", "vLLM2", "Create New"], value="vLLM1", show_label=False, info="Select a vLLM or create a new one. Where?") 
                
                    with gr.Column(scale=1):
                        with gr.Row() as vllm_prompt_output:
                            output_prompt = gr.Textbox(label="Prompt Output", lines=4, show_label=True)
                        with gr.Row() as vllm_prompt:
                            prompt_btn = gr.Button("PROMPT")

        
        
            with gr.TabItem("Audio", id=4):
                with gr.Row(visible=True) as row_vllm_audio:
                    with gr.Column(scale=2):
                        with gr.Accordion(("Automatic Speech Recognition"), open=True, visible=True) as acc_audio:
                            audio_input = gr.Audio(label="Upload Audio", type="filepath")
                            audio_model=gr.Dropdown(defaults_frontend['audio_models'], label="Model size", info="Select a Faster-Whisper model")
                            
                            device=gr.Radio(["cpu", "cuda"], value="cpu", label="Select architecture", info="Your system supports CUDA!. Make sure all drivers installed. /checkcuda if cuda")
                            compute_type=gr.Radio(["int8"], value="int8", label="Compute type", info="Select a compute type")
                            
                            translate_checkbox = gr.Checkbox(label="Enable Translation", value=False)
                            source_lang = gr.Dropdown(["auto"], value="auto", label="Source Language", info="Auto-detection by default")
                            target_lang = gr.Dropdown(["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"], 
                                                    value="en", label="Target Language")
                
                
                    with gr.Column(scale=1):
                        with gr.Row() as vllm_prompt_output:
                            audio_path = gr.Textbox(visible=True)
                            text_output = gr.Textbox(label="Transcription", lines=8)
                            srt_status = gr.Textbox(label="SRT Status", visible=True)
            
                        with gr.Row() as vllm_prompt:
                            transcribe_btn = gr.Button("Transcribe")
                            transcribe_srt_btn = gr.Button("Transcribe + SRT")
                            translate_btn = gr.Button("Translate", visible=True)
                            download_srt_btn = gr.Button("Download SRT", visible=True)
                            srt_download_link = gr.File(label="SRT File", visible=True)
        
        
        output = gr.Textbox(label="Output", lines=4, show_label=True, visible=True)     
        
        # aaaa
        

        # Audio processing functions
        def process_audio(audio_path, audio_model, device, compute_type, action):
            try:
                print(f'Processing audio with action: {action}')
                logging.info(f'Processing audio with action: {action}')
                
                response = requests.post(AUDIO_URL, json={
                    "method": "transcribe",
                    "audio_model": audio_model,
                    "audio_path": audio_path,
                    "device": device,
                    "compute_type": compute_type,
                    "generate_srt": (action == "transcribe_srt")
                }, timeout=REQUEST_TIMEOUT)

                if response.status_code == 200:
                    res_json = response.json()
                    if res_json["result_status"] != 200:
                        raise Exception(res_json["result_data"])
                    
                    result = res_json["result_data"]
                    output_text = f"Detected language: {result['language']}\n\n{result['text']}\n\nProcessing time: {result['processing_time']}"
                    
                    if action == "transcribe_srt" and "srt_download_url" in result:
                        return {
                            text_output: output_text,
                            srt_status: "SRT generated successfully",
                            download_srt_btn: gr.Button(visible=True),
                            srt_download_link: gr.File(visible=False),
                            translate_btn: gr.Button(visible=True)
                        }
                    return {
                        text_output: output_text,
                        translate_btn: gr.Button(visible=True)
                    }
                else:
                    raise Exception(f"Request Error: {response.text}")
            
            except Exception as e:
                logging.exception(f'Exception occurred: {e}')
                return {
                    text_output: f"Error: {str(e)}",
                    srt_status: "Error generating SRT" if action == "transcribe_srt" else ""
                }

        def translate_text(text, source_lang, target_lang):
            try:
                # Extract the actual text content (after language detection info)
                content = "\n\n".join(text.split("\n\n")[1:]) if "\n\n" in text else text
                
                response = requests.post(BACKEND_URL + "/translate", json={
                    "text": content,
                    "source_lang": source_lang,
                    "target_lang": target_lang
                }, timeout=REQUEST_TIMEOUT)

                if response.status_code == 200:
                    res_json = response.json()
                    if res_json["result_status"] != 200:
                        raise Exception(res_json["result_data"])
                    
                    translation = res_json["result_data"]["translated_text"]
                    return f"Translation ({res_json['result_data']['target_lang']}):\n\n{translation}"
                else:
                    raise Exception(f"Translation Error: {response.text}")
            
            except Exception as e:
                logging.exception(f'Translation error: {e}')
                return f"Translation Error: {str(e)}"

        def download_srt():
            try:
                # This would be implemented based on your backend's SRT file handling
                # You might need to store the SRT path temporarily or use a session-based approach
                return {
                    srt_download_link: gr.File(visible=True, value="path_to_srt.srt")
                }
            except Exception as e:
                return {
                    srt_status: f"Download Error: {str(e)}"
                }

                
        

        # transcribe_btn.click(
        #     get_audio_path,
        #     audio_input,
        #     [text_output,audio_path]
        #     ).then(
        #     transcribe_audio,
        #     [audio_model,audio_path,device,compute_type],
        #     text_output
        # )
        
        
        transcribe_btn.click(
            fn=process_audio,
            inputs=[audio_input, audio_model, device, compute_type, gr.Text("transcribe_only", visible=False)],
            outputs=[text_output, srt_status, download_srt_btn, srt_download_link, translate_btn]
        )

        transcribe_srt_btn.click(
            fn=process_audio,
            inputs=[audio_input, audio_model, device, compute_type, gr.Text("transcribe_srt", visible=False)],
            outputs=[text_output, srt_status, download_srt_btn, srt_download_link, translate_btn]
        )

        translate_btn.click(
            fn=translate_text,
            inputs=[text_output, source_lang, target_lang],
            outputs=[text_output]
        )

        download_srt_btn.click(
            fn=download_srt,
            outputs=[srt_download_link]
        )

        # Show/hide translation button based on checkbox
        translate_checkbox.change(
            fn=lambda x: gr.Button(visible=x),
            inputs=[translate_checkbox],
            outputs=[translate_btn]
        )

        # Update audio path when file is uploaded
        audio_input.change(
            fn=lambda x: x,
            inputs=[audio_input],
            outputs=[audio_path]
        )
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        kekw = gr.Textbox(label="kekw")

                


        with gr.Accordion(("System Stats"), open=False) as acc_system_stats:
            
            with gr.Accordion(("GPU information"), open=False) as acc_gpu_dataframe:
                gpu_dataframe = gr.Dataframe()

            with gr.Accordion(("Disk information"), open=False) as acc_disk_dataframe:
                disk_dataframe = gr.Dataframe()

            with gr.Accordion(("Network information"), open=False) as acc_network_dataframe:
                network_dataframe = gr.Dataframe()




        with gr.Column(scale=1, visible=True) as vllm_running_engine_argumnts_btn:
            vllm_running_engine_arguments_show = gr.Button("LOAD VLLM CREATEEEEEEEEUUUUHHHHHHHH", variant="primary")
            vllm_running_engine_arguments_close = gr.Button("CANCEL")

                





        
        btn_interface = gr.Button("Load Interface",visible=False)
        @gr.render(inputs=[selected_model_pipeline_tag, selected_model_id], triggers=[btn_interface.click])
        def show_split(text_pipeline, text_model):
            if len(text_model) == 0:
                gr.Markdown("Error pipeline_tag or model_id")
            else:
                selected_model_id_arr = str(text_model).split('/')
                print(f'selected_model_id_arr {selected_model_id_arr}...')            
                gr.Interface.from_pipeline(pipeline(text_pipeline, model=f'/models/{selected_model_id_arr[0]}/{selected_model_id_arr[1]}'))










        
        
        
        
        
        
        
        
        
        

        
        
        
       
        load_btn = gr.Button("Load into vLLM (port: 1370)", visible=True)

        









        test_vllm_radio_out = gr.Textbox(label="test_vllm_radio_out")

        # aaaa2
        test_vllms_radio = gr.Radio(["container_vllm_xoo", "container_vllm_oai", "Create New"], value="container_vllm_oai", show_label=False, info="Select a vllms_prompt or create a new one. Where?")

        kekt = gr.Textbox(label="kekt")
        


        for container in test_vllms_list_running:

            
            test_vllm_id = gr.Textbox(value=container["id"], elem_classes="table-cell", label="test_vllm_id")
            
            test_vllm_status = gr.Textbox(value=container["State"]["Status"], elem_classes="table-cell", label="test_vllm_status")
                                
            test_vllm_ts = gr.Textbox(value=container["ts"], elem_classes="table-cell", label="test_vllm_ts")
            
            


        # test_vllms_radio.change(
        #     toggle_test_vllms,
        #     test_vllms_radio,
        #     test_vllm_radio_out
        # )

        # vllms=gr.Radio(["vLLM1", "vLLM2", "Create New"], value="vLLM1", show_label=False, info="Select a vLLM or create a new one. Where?")
        # vllms.change(
        #     toggle_vllm_load_create,
        #     vllms,
        #     [acc_load, vllm_load_actions, acc_create, vllm_create_actions]
        # )

        # vllmsprompt.change(
        #     toggle_vllm_prompt,
        #     vllmsprompt,
        #     testtext
        # )



        
        
        
        
        
        
        current_tab.change(
            change_tab,
            current_tab,
            tabs
        )
        
        
        
        
        
        
        
        
        device.change(
            toggle_compute_type,
            device,
            [compute_type]
        )
        
        
        
        
        
        input_search.change(
            search_change,
            input_search,
            [model_dropdown,input_search],
            show_progress=False
        )
        
        input_search.submit(
            search_models, 
            input_search, 
            [model_dropdown,input_search]
        ).then(
            lambda: gr.update(visible=True),
            None, 
            model_dropdown
        )
        
        btn_search.click(
            search_models, input_search, 
            [model_dropdown,input_search]
        ).then(
            lambda: gr.update(visible=True),
            None,
            model_dropdown
        )

        btn_tested_models.click(
            dropdown_load_tested_models,
            None,
            [model_dropdown,input_search]
        )




        
        # input_search.submit(search_models, inputs=input_search, outputs=[model_dropdown,input_search]).then(lambda: gr.update(visible=True), None, model_dropdown)
        # btn_search.click(search_models, inputs=input_search, outputs=[model_dropdown,input_search]).then(lambda: gr.update(visible=True), None, model_dropdown)







        

        btn_dl.click(
            parallel_download, 
            [selected_model_size, model_dropdown], 
            download_status,
            concurrency_limit=15
        )


        btn_dl.click(
            lambda: gr.update(label="Starting download",visible=True),
            None,
            download_status
        ).then(
            download_info, 
            selected_model_size,
            download_status,
            concurrency_limit=15
        ).then(
            download_from_hf_hub, 
            model_dropdown,
            download_status,
            concurrency_limit=15
        ).then(
            lambda: gr.update(label="Download finished"),
            None,
            download_status
        ).then(
            lambda: gr.update(visible=False),
            None,
            row_download
        ).then(
            lambda: gr.update(visible=True),
            None,
            btn_interface
        ).then(
            lambda: gr.update(visible=True),
            None,
            acc_load
        ).then(
            lambda: gr.update(visible=True),
            None,
            vllm_load_actions
        ).then(
            lambda: gr.update(visible=True, open=False),
            None,
            acc_load
        )






        input_search.submit(
            search_models, 
            input_search, 
            [model_dropdown]
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            model_dropdown
        )
        
        btn_search.click(
            search_models, 
            input_search, 
            [model_dropdown]
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            model_dropdown
        )




        model_dropdown.change(
            get_info, 
            model_dropdown, 
            [selected_model_search_data,selected_model_id,selected_model_architectures,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_downloads,selected_model_container_name]
        ).then(
            get_additional_info, 
            model_dropdown, 
            [selected_model_hf_data, selected_model_config_data, selected_model_architectures,selected_model_id, selected_model_size, selected_model_gated, selected_model_model_type, selected_model_quantization, selected_model_torch_dtype, selected_model_hidden_size]
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            row_model_select
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            row_model_info
        ).then(
            gr_load_check, 
            [selected_model_id, selected_model_architectures, selected_model_pipeline_tag, selected_model_transformers, selected_model_size, selected_model_private, selected_model_gated, selected_model_model_type, selected_model_quantization],
            [output,row_download,btn_load]
        )


        vllm_running_engine_arguments_show.click(
            lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], 
            None, 
            [vllm_running_engine_arguments_show, vllm_running_engine_arguments_close, acc_load]
        )
        
        vllm_running_engine_arguments_close.click(
            lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], 
            None, 
            [vllm_running_engine_arguments_show, vllm_running_engine_arguments_close, acc_load]
        )




        btn_load.click(
            lambda: gr.update(label="Deploying"),
            None,
            output
        ).then(
            lambda: gr.update(visible=True, open=False), # hier
            None, 
            acc_load    
        ).then(
            llm_load,
            vllm_load_components.to_list(),
            [output]
        ).then(
            lambda: gr.update(visible=True, open=True), 
            None, 
            acc_prompt
        ).then(
            lambda: gr.update(visible=True), # hier
            None, 
            btn_load
        ).then(
            lambda: gr.update(visible=True, open=True), 
            None, 
            acc_prompt
        ).then(
            refresh_container,
            [container_state]
        )

        btn_create.click(
            lambda: gr.update(label="Deploying"),
            None,
            output
        ).then(
            lambda: gr.update(visible=True, open=False), 
            None, 
            acc_create    
        ).then(
            llm_create,
            vllm_create_components.to_list(),
            [output]
        ).then(
            lambda: gr.update(visible=True, open=True),
            None, 
            acc_prompt
        ).then(
            lambda: gr.update(visible=True), # hier
            None, 
            btn_create
        ).then(
            lambda: gr.update(visible=True), # hier
            None, 
            btn_create_close
        ).then(
            refresh_container,
            [container_state]
        )


        
        prompt_btn.click(
            llm_prompt,
            llm_prompt_components.to_list(),
            [output_prompt]
        )


        vllms.change(
            toggle_vllm_load_create,
            vllms,
            [acc_load, vllm_load_actions, acc_create, vllm_create_actions]
        )

        
        
        
        # aaaa label info evtl oder info
        test_vllms_radio.change(
            toggle_test_vllms,
            test_vllms_radio,
            [test_vllms_radio,test_vllm_radio_out] #choices und ts
        )

                


        disk_timer = gr.Timer(1,active=True)
        disk_timer.tick(disk_to_pd, outputs=disk_dataframe)

        gpu_timer = gr.Timer(1,active=True)
        gpu_timer.tick(gpu_to_pd, outputs=gpu_dataframe)

        network_timer = gr.Timer(1,active=True)
        network_timer.tick(network_to_pd, outputs=[network_dataframe,kekw])


    return app

# Launch the app
if __name__ == "__main__":
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [start] starting frontend ...')
    logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [start] starting frontend ...')
    
    backend_url = f'http://container_backend:{os.getenv("BACKEND_PORT")}/docker'
    
    # Wait for the backend container to be online
    if wait_for_backend(backend_url):
                
        print(f' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ OK CALLING BACKEND TO GET VLLMS')
        req_test = {
            "method": "test"
        }

        
        print(f' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 1')
        res_vllms = redis_api("test", req_dict=req_test)
        print(f' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 2')
        print(f' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ OK GOT res_vllms: {res_vllms}')
        print(f' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ OK GOT res_vllms[0]: {res_vllms[0]}')
        print(f' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ OK GOT res_vllms[1]: {res_vllms[1]}')

        print(f' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 3')

        app = create_app()
        app.launch(server_name=f'{os.getenv("FRONTEND_IP")}', server_port=int(os.getenv("FRONTEND_PORT")))
    else:
        print(f'Failed to start application due to backend container not being online.')
