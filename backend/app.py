from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import json
import docker
from docker.types import DeviceRequest
import time
import os
import requests
import redis.asyncio as redis
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
import pynvml
import psutil
import logging




# print(f'** connecting to redis on port: {os.getenv("REDIS_PORT")} ... ')
r = redis.Redis(host="redis", port=int(os.getenv("REDIS_PORT", 6379)), db=0)

LOG_PATH = './logs'
LOGFILE_CONTAINER = f'{LOG_PATH}/logfile_container_backend.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.INFO, 
                   format='[%(asctime)s - %(name)s - %(levelname)s - %(message)s]')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] started logging in {LOGFILE_CONTAINER}')
print(f'** connecting to pynvml ... ')
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()
print(f'** pynvml found GPU: {device_count}')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] pynvml found GPU: {device_count}')

device_uuids = []
for i in range(0,device_count):
    # print(f'1 i {i}')
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    # print(f'1 handle {handle}')
    current_uuid = pynvml.nvmlDeviceGetUUID(handle)
    device_uuids.append(current_uuid)

# print(f'** pynvml found uuids ({len(device_uuids)}): {device_uuids} ')


DEFAULTS_PATH = "/usr/src/app/utils/defaults.json"
if not os.path.exists(DEFAULTS_PATH):
    logging.info(f' [START] File missing: {DEFAULTS_PATH}')

with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
    defaults_backend = json.load(f)["backend"]
    logging.info(f' [START] SUCCESS! Loaded: {DEFAULTS_PATH}')
    DEFAULT_CONTAINER_STATS = defaults_backend['DEFAULT_CONTAINER_STATS']
    logging.info(f' [START] SUCCESS! Loaded DEFAULT_CONTAINER_STATS: {DEFAULT_CONTAINER_STATS}')
    COMPUTE_CAPABILITIES = defaults_backend['compute_capability']
    logging.info(f' [START] SUCCESS! Loaded COMPUTE_CAPABILITIES: {COMPUTE_CAPABILITIES}')





# created (running time)
# port 
# gpu name 
# gpu uuid
# public or private 
# user 
# model 
# vllm image 
# prompts amount
# tokens

# computed





async def save_redis(**kwargs):
    try:
        if not kwargs:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [save_redis] [error] No data')
            return f'no data'
        else:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [save_redis] kwargs: {kwargs}')
        if not kwargs["db_name"]:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [save_redis] [error] No db_name')
            return f'no db_name'
                
        if not 'vllm_id' in kwargs:            
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [save_redis] No vllm_id provided. Creating new ...')
            vllm_id = f'vllm_{str(int(datetime.now().timestamp()))}'
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [save_redis] ... vllm_id: {vllm_id}')        
        
        res_db_list = r.lrange(kwargs["db_name"], 0, -1)
        if len(res_db_list) > 0:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [save_redis] found {len(res_db_list)} entries!')
            vllm_id_list = [entry for entry in res_db_list if json.loads(entry)["vllm_id"] == kwargs["vllm_id"]]
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [save_redis] found {len(vllm_id_list)} for {kwargs["vllm_id"]}')
            
            if len(vllm_id_list) > 0:
                print(f'Found {kwargs["vllm_id"]}! Updating')                
                for entry in res_db_list:
                    parsed_entry = json.loads(entry)  # Convert JSON string to dictionary
                    print(f'*** parsed_entry {parsed_entry["vllm_id"]}!')
                    if parsed_entry["vllm_id"] == kwargs["vllm_id"]:
                        print(f'found vllm_id {kwargs["vllm_id"]}!')
                        r.lrem(kwargs["db_name"], 0, entry)
                        print("entry deleted!")
                        parsed_entry['ts'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print("trying push ...")
                        r.rpush(kwargs["db_name"], json.dumps(parsed_entry))
                        print("pushed!")
            else:                
                print(f'didnt find {kwargs["vllm_id"]} yet! Creating')
                redis_data = {
                    "db_name": kwargs["db_name"],
                    "vllm_id": kwargs["vllm_id"],
                    "model": kwargs["model"], 
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }    
                r.rpush(kwargs["db_name"], json.dumps(redis_data))
                print("created!")

        else:
            print("no entry found yet .. creating")
            redis_data = {
                "db_name": kwargs["db_name"],
                "vllm_id": vllm_id,
                "model": kwargs["model"], 
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            r.rpush(kwargs["db_name"], json.dumps(redis_data))
            print("created!")

    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [save_redis] [error]: {e}')







prev_bytes_recv = 0
def get_download_speed():
    try:
        global prev_bytes_recv
        net_io = psutil.net_io_counters()
        bytes_recv = net_io.bytes_recv
        download_speed = bytes_recv - prev_bytes_recv
        prev_bytes_recv = bytes_recv
        download_speed_kb = download_speed / 1024
        download_speed_mbit_s = (download_speed * 8) / (1024 ** 2)      
        bytes_received_mb = bytes_recv
        return f'download_speed_mbit_s {download_speed_mbit_s} bytes_recv {bytes_recv} download_speed {download_speed} download_speed_kb {download_speed_kb} '
        # return f'{download_speed_kb:.2f} KB/s (total: {bytes_received_mb:.2f})'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'download error: {e}'



def get_network_info():
    network_info = []
    try: 
        current_total_dl = get_download_speed()
        network_info.append({
            "container": f'all',
            "info": "infoblabalba",            
            "current_dl": f'{current_total_dl}',
            "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        })
        res_container_list = client.containers.list(all=True)
        for container in res_container_list:
            container_stats = container.stats(stream=False)
            networks = container_stats.get('networks', {})
            rx_bytes = 0
            if networks:
                rx_bytes = sum(network.get('rx_bytes', 0) for network in networks.values())

            network_info.append({
                "container": container.name,
                "info": "infoblabalba",
                "current_dl": str(rx_bytes),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
   
        return network_info
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_network_info] {e}')
        return network_info

async def redis_timer_network():
    while True:
        try:
            current_network_info = get_network_info()
            res_db_network = await r.get('db_network')
            if res_db_network is not None:
                db_network = json.loads(res_db_network)
                updated_network_data = []
                for net_info_obj in current_network_info:
                    update_data = {
                        "container": str(net_info_obj["container"]),
                        "info": str(net_info_obj["info"]),
                        "current_dl": str(net_info_obj["current_dl"]),
                        "timestamp": str(net_info_obj["timestamp"]),
                    }
                    updated_network_data.append(update_data)
                await r.set('db_network', json.dumps(updated_network_data))
            else:
                updated_network_data = []
                for net_info_obj in current_network_info:
                    update_data = {
                        "container": str(net_info_obj["container"]),
                        "info": str(net_info_obj["info"]),
                        "current_dl": str(net_info_obj["current_dl"]),
                        "timestamp": str(net_info_obj["timestamp"]),
                    }
                    updated_network_data.append(update_data)
                    # print(f'[network] 2 updated_network_data: {updated_network_data}')
                await r.set('db_network', json.dumps(updated_network_data))
            await asyncio.sleep(1.0)
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error: {e}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [redis_timer_network] {e}')
            await asyncio.sleep(1.0)





def get_disk_info():
    try:
        disk_info = []
        partitions = psutil.disk_partitions(all=False)
        processed_devices = set()
        for partition in partitions:
            device = partition.device
            if device not in processed_devices:
                processed_devices.add(device)
                current_disk_info = {}
                try:                
                    current_disk_info['device'] = str(partition.device)
                    current_disk_info['mountpoint'] = str(partition.mountpoint)
                    current_disk_info['fstype'] = str(partition.fstype)
                    current_disk_info['opts'] = str(partition.opts)
                    
                    disk_usage = psutil.disk_usage(partition.mountpoint)
                    current_disk_info['usage_total'] = f'{disk_usage.total / (1024**3):.2f} GB'
                    current_disk_info['usage_used'] = f'{disk_usage.used / (1024**3):.2f} GB'
                    current_disk_info['usage_free'] = f'{disk_usage.free / (1024**3):.2f} GB'
                    current_disk_info['usage_percent'] = f'{disk_usage.percent}%'
                    
                except Exception as e:
                    print(f'[ERROR] [get_disk_info] Usage: [Permission denied] {e}')
                    pass
                
                try:                
                    io_stats = psutil.disk_io_counters()
                    current_disk_info['io_read_count'] = str(io_stats.read_count)
                    current_disk_info['io_write_count'] = str(io_stats.write_count)
                    
                except Exception as e:
                    print(f'[ERROR] [get_disk_info] Disk I/O statistics not available on this system {e}')
                    pass
                
                disk_info.append({                
                    "device": current_disk_info.get("device", "0"),
                    "mountpoint": current_disk_info.get("mountpoint", "0"),
                    "fstype": current_disk_info.get("fstype", "0"),
                    "opts": current_disk_info.get("opts", "0"),
                    "usage_total": current_disk_info.get("usage_total", "0"),
                    "usage_used": current_disk_info.get("usage_used", "0"),
                    "usage_free": current_disk_info.get("usage_free", "0"),
                    "usage_percent": current_disk_info.get("usage_percent", "0"),
                    "io_read_count": current_disk_info.get("io_read_count", "0"),
                    "io_write_count": current_disk_info.get("io_write_count", "0")
                })

        return disk_info
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_disk_info] [ERROR] e -> {e}')
        return f'{e}'

total_disk_info = get_disk_info()

async def redis_timer_disk():
    while True:
        try:
            total_disk_info = get_disk_info()
            res_db_disk = await r.get('db_disk')
            if res_db_disk is not None:
                db_disk = json.loads(res_db_disk)
                updated_disk_data = []
                for disk_i in range(0,len(total_disk_info)):
                    update_data = {
                        "disk_i": disk_i,
                        "disk_info": str(total_disk_info[disk_i]),
                        "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    }
                    updated_disk_data.append(update_data)
                await r.set('db_disk', json.dumps(updated_disk_data))
            else:
                updated_disk_data = []
                for disk_i in range(0,len(total_disk_info)):
                    update_data = {
                        "disk_i": disk_i,
                        "disk_info": str(total_disk_info[disk_i]),
                        "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    }
                    updated_disk_data.append(update_data)
                await r.set('db_disk', json.dumps(updated_disk_data))
            await asyncio.sleep(1.0)
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error: {e}')
            await asyncio.sleep(1.0)



def get_gpu_info():
    try:

        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info = []
        for i in range(0,device_count):
            current_gpu_info = {}
            current_gpu_info['res_gpu_i'] = str(i)           
            

            
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            

            
            try:
                res_uuid = pynvml.nvmlDeviceGetUUID(handle)
                current_gpu_info['res_uuid'] = f'{res_uuid}'
            except Exception as e:
                print(f'0 gpu_info {e}')
                current_gpu_info['res_uuid'] = f'0'
            
            
            
            try:
                res_name = pynvml.nvmlDeviceGetName(handle)
                current_gpu_info['res_name'] = f'{res_name}'
            except Exception as e:
                print(f'00 gpu_info {e}')
                current_gpu_info['res_name'] = f'0'
            
            
            
        
            
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                # mem_util = f'{(mem_used / mem_total) * 100} %'
                res_gpu_util = f'{utilization.gpu}%'
                current_gpu_info['res_gpu_util'] = f'{res_gpu_util}'
                
                
                # res_mem_util = f'{utilization.memory}%'
                # current_gpu_info['res_mem_util'] = f'{res_mem_util}'
            except Exception as e:
                print(f'1 gpu_info {e}')

            try: 
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                res_mem_total = f'{mem_info.total / 1024 ** 2:.2f} MB'
                current_gpu_info['res_mem_total'] = f'{res_mem_total}'
                res_mem_used = f'{mem_info.used / 1024 ** 2:.2f} MB'
                current_gpu_info['res_mem_used'] = f'{res_mem_used}'
                res_mem_free = f'{mem_info.free / 1024 ** 2:.2f} MB'
                current_gpu_info['res_mem_free'] = f'{res_mem_free}'
                
                res_mem_util = (float(mem_info.used / 1024**2)/float(mem_info.total / 1024**2)) * 100
                current_gpu_info['res_mem_util'] = f'{"{:.2f}".format(res_mem_util)}% ({res_mem_used}/{res_mem_total})'

            except Exception as e:
                print(f'2 gpu_info {e}')
            
            try:
                # Get GPU temperature
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                res_temperature = f'{temperature}°C'
                current_gpu_info['res_temperature'] = f'{res_temperature}'
            except Exception as e:
                print(f'3 gpu_info {e}')
                
            try:
                # Get GPU fan speed
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                res_fan_speed = f'{fan_speed}%'
                current_gpu_info['res_fan_speed'] = f'{res_fan_speed}'
            except Exception as e:
                print(f'4 gpu_info {e}')


            try:
                # Get GPU power usage
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)
                res_power_usage = f'{power_usage / 1000:.2f} W'
                current_gpu_info['res_power_usage'] = f'{res_power_usage}'
            except Exception as e:
                print(f'5 gpu_info {e}')
        
        
            try:
                # Get GPU clock speeds
                clock_info_graphics = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                res_clock_info_graphics = f'{clock_info_graphics} MHz'
                current_gpu_info['res_clock_info_graphics'] = f'{res_clock_info_graphics}'
            except Exception as e:
                print(f'6 gpu_info {e}')
            
            
            try:
                clock_info_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                res_clock_info_mem = f'{clock_info_mem} MHz'
                current_gpu_info['res_clock_info_mem'] = f'{res_clock_info_mem}'
            except Exception as e:
                print(f'7 gpu_info {e}')
                
            try:
                # Get GPU compute capability (compute_capability)
                cuda_cores = pynvml.nvmlDeviceGetNumGpuCores(handle)
                res_cuda_cores = f'{cuda_cores}'
                current_gpu_info['res_cuda_cores'] = f'{res_cuda_cores}'
            except Exception as e:
                print(f'8 gpu_info {e}')

            res_supported = []
            res_not_supported = []
            try:
                # Get GPU compute capability (CUDA cores)
                compute_capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability_str = f'{compute_capability[0]}.{compute_capability[1]}'
                res_compute_capability = f'{compute_capability_str}'

                if float(res_compute_capability) >= 8:
                    res_supported.append('Bfloat16')
                else:
                    res_not_supported.append('Bfloat16')
            except Exception as e:
                print(f'9 gpu_info {e}')
                res_compute_capability = 0

            if res_compute_capability == 0:
                try:
                    res_name = pynvml.nvmlDeviceGetName(handle)
                    res_name_split = res_name.split(" ")[1:]
                    res_name_splitted_str = " ".join(res_name_split)
                    if res_name.lower() in defaults_backend['compute_capability']:
                        print(f'-> res_name {res_name} exists with compute capability {defaults_backend["compute_capability"][res_name.lower()]}')
                        res_compute_capability = f'{defaults_backend["compute_capability"][res_name.lower()]}'
                    elif res_name_splitted_str.lower() in defaults_backend['compute_capability']:
                        print(f'-> res_name_splitted_str {res_name_splitted_str} exists with compute capability {defaults_backend["compute_capability"][res_name.lower()]}')
                        res_compute_capability = f'{defaults_backend["compute_capability"][res_name_splitted_str.lower()]}'
                    else:
                        print(f'{res_name.lower()} or {res_name_splitted_str.lower()} not found in database')
                except Exception as e:
                    print(f'99 res_compute_capability e: {e}')

            
            
            res_supported_str = ",".join(res_supported)
            current_gpu_info['res_supported_str'] = f'{res_supported_str}'
            res_not_supported_str = ",".join(res_not_supported)
            current_gpu_info['res_not_supported_str'] = f'{res_not_supported_str}'
            
            gpu_info.append({                
                "gpu_i": current_gpu_info.get("res_gpu_i", "0"),
                "name": current_gpu_info.get("res_name", "0"),
                "current_uuid": current_gpu_info.get("res_uuid", "0"),
                "gpu_util": current_gpu_info.get("res_gpu_util", "0"),
                "mem_util": current_gpu_info.get("res_mem_util", "0"),
                "mem_total": current_gpu_info.get("res_mem_total", "0"),
                "mem_used": current_gpu_info.get("res_mem_used", "0"),
                "mem_free": current_gpu_info.get("res_mem_free", "0"),
                "temperature": current_gpu_info.get("res_temperature", "0"),
                "fan_speed": current_gpu_info.get("res_fan_speed", "0"),
                "power_usage": current_gpu_info.get("res_power_usage", "0"),
                "clock_info_graphics": current_gpu_info.get("res_clock_info_graphics", "0"),
                "clock_info_mem": current_gpu_info.get("res_clock_info_mem", "0"),
                "cuda_cores": current_gpu_info.get("res_cuda_cores", "0"),
                "compute_capability": current_gpu_info.get("res_compute_capability", "0"),
                "supported": current_gpu_info.get("res_supported", "0"),
                "not_supported": current_gpu_info.get("res_not_supported", "0"),
                "not_supported": current_gpu_info.get("res_not_supported", "0")
            })
                        
        return gpu_info
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'{e}'



total_gpu_info = get_gpu_info()

async def redis_timer_gpu():
    while True:
        try:
            total_gpu_info = get_gpu_info()
            res_db_gpu = await r.get('db_gpu')
            if res_db_gpu is not None:
                db_gpu = json.loads(res_db_gpu)
                updated_gpu_data = []
                for gpu_i in range(0,len(total_gpu_info)):
                    update_data = {
                        "gpu_i": gpu_i,
                        "gpu_info": str(total_gpu_info[gpu_i]),
                        "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    }
                    updated_gpu_data.append(update_data)
                await r.set('db_gpu', json.dumps(updated_gpu_data))
            else:
                updated_gpu_data = []
                for gpu_i in range(0,len(total_gpu_info)):
                    update_data = {
                        "gpu_i": gpu_i,
                        "gpu_info": str(total_gpu_info[gpu_i]),
                        "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    }
                    updated_gpu_data.append(update_data)
                await r.set('db_gpu', json.dumps(updated_gpu_data))
            await asyncio.sleep(1.0)
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error: {e}')
            await asyncio.sleep(1.0)




# created (running time)
# port 
# gpu name 
# gpu uuid
# public or private 
# user 
# model 
# vllm image 
# prompts amount
# tokens

# computed

# aaaaa


        
def update_redis(**kwargs):
    if not kwargs:
        print(f'[update_redis] No data')
        return False
    if not 'db_name' in kwargs:
        print(f'[update_redis]  no db_name')
        return False
    else:
        print(f'kwargs["db_name"]: {kwargs["db_name"]}')
    if not kwargs["db_name"]:
        print(f'[update_redis] Error: Missing "db_name" in input data')
        return False
    
    print(f'[update_redis] getting db ...')
    res_db_list = r.lrange(kwargs["db_name"], 0, -1)
    print(f'[update_redis] got db: {res_db_list}')
    
    print(f'HÄÄÄÄÄÄ 1')
    if res_db_list:
        print(f'HÄÄÄÄÄÄ 2')
        print(f'found {len(res_db_list)} entries!')
        
        req_vllm_id_list = [entry for entry in res_db_list if json.loads(entry)["vllm_id"] == kwargs["vllm_id"]]
        print(f'found req_vllm_id_list {req_vllm_id_list}!')
        print(f'found req_vllm_id_list {len(req_vllm_id_list)}!')
        
        if len(req_vllm_id_list) > 0:
            print(f'Found {kwargs["vllm_id"]}! Updating')
            for entry in res_db_list:
                parsed_entry = json.loads(entry)  # Convert JSON string to dictionary
                print(f'*** parsed_entry {parsed_entry["vllm_id"]}!')
                if parsed_entry["vllm_id"] == kwargs["vllm_id"]:
                    print(f'found vllm_id {kwargs["vllm_id"]}!')
                    r.lrem(kwargs["db_name"], 0, entry)
                    print("entry deleted!")
                    parsed_entry['ts'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print("trying push ...")
                    r.rpush(kwargs["db_name"], json.dumps(parsed_entry))
                    print("pushed!")
                    return True
        else:
            print(f'didnt find {kwargs["vllm_id"]} yet! Creating')
            update_data1 = {
                "db_name": kwargs["db_name"],
                "vllm_id": kwargs["vllm_id"],
                "container": "b", 
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }        
            r.rpush(kwargs["db_name"], json.dumps(update_data1))
            print("created!")
            return True
    else:
        print(f'HÄÄÄÄÄÄ ELSEEEEEEE')
        print("no entry found yet .. creating")
        vllm_id = f'vllm_{str(int(datetime.now().timestamp()))}'
        update_data1 = {
            "db_name": kwargs["db_name"],
            "vllm_id": kwargs["vllm_id"],
            "container": "b", 
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }        
        r.rpush(kwargs["db_name"], json.dumps(update_data1))
        print("created!")
        return True
    return True




def get_vllm_info():
    try:        

        # print(f' @@@ [get_vllm_info] testing redis ...!')
        print(f' @@@ [get_vllm_info] VOR ERROR??????????????????????????????????????????????????')
        update_data3 = {"db_name": "db_vllm", "vllm_id": "10", "container": "b", "ts": f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'}
        res_update_redis = update_redis(**update_data3)
        print(f' @@@ [get_vllm_info] res_update_redis: {res_update_redis} ...!')
        
        
        # redis_data = {"db_name": "db_vllm", "vllm_id": "10", "model": "blabla", "ts": "123"}
        # print(f' @@@ [get_vllm_info] trying to save redis ...')
        # save_redis(**redis_data)
        # print(f' @@@ [get_vllm_info] saved redis!')

        return f'{update_data3} saved!'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_vllm_info] {e}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_vllm_info] [ERROR] e -> {e}')
        return f'{e}'


total_vllm_info = get_vllm_info()

async def redis_timer_vllm2():
    while True:
        try:
            current_network_info = get_network_info()
            res_db_network = await r.get('db_vllm2')
            if res_db_network is not None:
                db_network = json.loads(res_db_network)
                updated_network_data = []
                for net_info_obj in current_network_info:
                    update_data = {
                        "container": str(net_info_obj["container"]),
                        "info": str(net_info_obj["info"]),
                        "current_dl": str(net_info_obj["current_dl"]),
                        "timestamp": str(net_info_obj["timestamp"]),
                    }
                    updated_network_data.append(update_data)
                await r.set('db_vllm2', json.dumps(updated_network_data))
            else:
                updated_network_data = []
                for net_info_obj in current_network_info:
                    update_data = {
                        "container": str(net_info_obj["container"]),
                        "info": str(net_info_obj["info"]),
                        "current_dl": str(net_info_obj["current_dl"]),
                        "timestamp": str(net_info_obj["timestamp"]),
                    }
                    updated_network_data.append(update_data)
                    # print(f'[network] 2 updated_network_data: {updated_network_data}')
                await r.set('db_vllm2', json.dumps(updated_network_data))
            await asyncio.sleep(1.0)
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error: {e}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [redis_timer_network] {e}')
            await asyncio.sleep(1.0)

async def redis_timer_vllm():
    while True:
        try:
            total_vllm_info = get_vllm_info()
            res_db_vllm = await r.get('db_vllm')
            if res_db_vllm is not None:
                db_vllm = json.loads(res_db_vllm)
                updated_vllm_data = []
                for vllm_i in range(0,len(total_vllm_info)):
                    update_data = {
                        "db_name": vllm_i,
                        "vllm_id": vllm_i,
                        "model": "redis_timer_vllm",
                        "ts": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    }
                    updated_vllm_data.append(update_data)
                await r.set('db_vllm', json.dumps(updated_vllm_data))
            else:
                updated_vllm_data = []
                for vllm_i in range(0,len(total_vllm_info)):
                    update_data = {
                        "db_name": vllm_i,
                        "vllm_id": vllm_i,
                        "model": "redis_timer_vllm",
                        "ts": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    }
                    updated_vllm_data.append(update_data)
                await r.set('db_vllm', json.dumps(updated_vllm_data))
            await asyncio.sleep(1.0)
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error: {e}')
            await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(redis_timer_gpu())
    asyncio.create_task(redis_timer_disk())
    asyncio.create_task(redis_timer_network())
    asyncio.create_task(redis_timer_vllm())
    asyncio.create_task(redis_timer_vllm2())
    yield

app = FastAPI(lifespan=lifespan)

print(f' %%%%% trying to start docker ...')
client = docker.from_env()
print(f' %%%%% docker started!')
print(f' %%%%% trying to docker network ...')
network_name = "sys_net"
# try:
#     network = client.networks.get(network_name)
# except docker.errors.NotFound:
#     network = client.networks.create(network_name, driver="bridge")
# print(f' %%%%% docker network started! ...')



device_request = DeviceRequest(count=-1, capabilities=[["gpu"]])




async def stop_vllm_container():
    try:
        print(f' -> stop_vllm_container')
        res_container_list = client.containers.list(all=True)
        print(f'-> mhmmhmhmh 1')
        vllm_containers_running = [c for c in res_container_list if c.name.startswith("container_vllm") and c.status == "running"]
        print(f'-> found total vLLM running containers: {len(vllm_containers_running)}')
        while len(vllm_containers_running) > 0:
            print(f'stopping all vLLM containers...')
            for vllm_container in vllm_containers_running:
                print(f'-> stopping container {vllm_container.name}...')
                vllm_container.stop()
                vllm_container.wait()
            res_container_list = client.containers.list(all=True)
            vllm_containers_running = [c for c in res_container_list if c.name.startswith("vllm") and c.status == "running"]
        print(f'-> all vLLM containers stopped successfully')
        return 200
    except Exception as e:
        print(f'-> error e: {e}') 
        return 500




























def redis_connection(**kwargs):
    try:
        
        print(f' ======== BACKEND HMMM [redis_connection] 1')
        if not kwargs:
            print(f' **REDIS: Error: no kwargs')
            return False
        # else:
        #     print(f' **REDIS: kwargs: {kwargs}')
        print(f' ======== BACKEND HMMM [redis_connection] 11 kwargs: {kwargs}')
        print(f' ======== BACKEND HMMM [redis_connection] 2')
        if not kwargs["db_name"]:
            print(f' **REDIS: Error: no db_name')
            return False
        print(f' ======== BACKEND HMMM [redis_connection] 3')
        if not kwargs["method"]:
            print(f' **REDIS: Error: no method')
            return False
        print(f' ======== BACKEND HMMM [redis_connection] 4')
        if not kwargs["select"]:
            print(f' **REDIS: Error: no select')
            return False
        print(f' ======== BACKEND HMMM [redis_connection] 5')
        res_db_list = r.lrange(kwargs["db_name"], 0, -1)
        print(f' ======== BACKEND HMMM [redis_connection] 6')
        print(f' ======== BACKEND HMMM [redis_connection] 6 res_db_list: {res_db_list}')
        print(f' ======== BACKEND HMMM [redis_connection] 6 found {len(res_db_list)} entries!')
        res_db_list = [json.loads(entry) for entry in res_db_list]
        print(f' ======== BACKEND HMMM [redis_connection] 7')
        
        if kwargs["select"] == "filter":
            if not kwargs["filter_key"]:
                print(f' **REDIS: Error: no filter_key')
                return False
            
            if not kwargs["filter_val"]:
                print(f' **REDIS: Error: no filter_val')
                return False

            res_db_list = [entry for entry in res_db_list if entry[kwargs["filter_key"]] == kwargs["filter_val"]]
            # print(f' **REDIS: filtered: {len(res_db_list)}')
        print(f' ======== BACKEND HMMM [redis_connection] 8')
        if kwargs["method"] == "get":
            print(f' ======== BACKEND HMMM [redis_connection] 9')
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
                    # entry["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    entry["gpu"]["mem"] = f'blablabla + {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
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




req_db = "db_test28"











                    
@app.get("/")
async def root():
    return f'Hello from server {os.getenv("BACKEND_PORT")}!'



@app.post("/redis")
async def fnredis(request: Request):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] >>>> [redis] START')
    try:
        req_data = await request.json()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] >>>> [redis] req_data > {req_data}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] >>>> [redis] req_data > {req_data}')
        
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] >>>> [redis] req_data["method"] > {req_data["method"]}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] >>>> [redis] req_data["method"] > {req_data["method"]}')
        
        
        # if req_data["method"] == "list":
        #     res_container_list = client.containers.list(all=True)
        #     return JSONResponse([container.attrs for container in res_container_list])
        if req_data["method"] == "test":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] >>>> [redis] req_data["method"] == "test"')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] >>>> [redis] req_data["method"] == "test"')
            vllm1 = {
                "container_name": "vllm1",
                "uid": "123123",
                "status": "running",
                "State": {
                    "Status": "running"
                },
                "gpu": {
                    "mem": "ok%"
                },
                "ts": "0"
            }

            vllm2 = {
                "container_name": "vllm2",
                "uid": "42124124",
                "status": "running",
                "State": {
                    "Status": "running"
                },
                "gpu": {
                    "mem": "ok%"
                },
                "ts": "0"
            }

            vllm3 = {
                "container_name": "vllm3",
                "uid": "523235235",
                "status": "running",
                "State": {
                    "Status": "running"
                },
                "gpu": {
                    "mem": "ok%"
                },
                "ts": "0"
            }

            vllm4 = {
                "container_name": "vllm4",
                "uid": "52352352",
                "status": "running",
                "State": {
                    "Status": "running"
                },
                "gpu": {
                    "mem": "ok%"
                },
                "ts": "0"
            }

            vllm5 = {
                "container_name": "vllm5",
                "uid": "74545",
                "status": "running",
                "State": {
                    "Status": "running"
                },
                "gpu": {
                    "mem": "ok%"
                },
                "ts": "0"
            }
            res_data = [vllm1,vllm2,vllm3,vllm4,vllm5]
            return JSONResponse({"result_status": 200, "result_data": res_data})
  
        if req_data["method"] == "vllm":
            
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] >>>> [redis] method == vllm ... trying request database ....')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] >>>> [redis] method == vllm ... trying request database ....')
            
            
            test_call_get = {
                "db_name": req_db,
                "method": "get",
                "select": "all"
            }
            
            res_vllm_list = redis_connection(**test_call_get)
            
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] >>>> [redis] returning ... res_vllm_list: {res_vllm_list}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] >>>> [redis] returning ... res_vllm_list: {res_vllm_list}')
            
            
            return JSONResponse({"result_status": 200, "result_data": res_vllm_list})
            # return JSONResponse(res_vllm_list)
            



    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": f'{e}'})


@app.post("/docker")
async def fndocker(request: Request):
    try:
        req_data = await request.json()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] req_data > {req_data}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] req_data > {req_data}')
        
        if req_data["method"] == "test":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] >>>> [redis] req_data["method"] == "test"')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] >>>> [redis] req_data["method"] == "test"')
            vllm1 = {
                "container_name": "vllm1",
                "uid": "123123",
                "status": "running",
                "State": {
                    "Status": "running"
                },
                "gpu": {
                    "mem": "ok%"
                },
                "ts": "0"
            }

            vllm2 = {
                "container_name": "vllm2",
                "uid": "42124124",
                "status": "running",
                "State": {
                    "Status": "running"
                },
                "gpu": {
                    "mem": "ok%"
                },
                "ts": "0"
            }

            vllm3 = {
                "container_name": "vllm3",
                "uid": "523235235",
                "status": "running",
                "State": {
                    "Status": "running"
                },
                "gpu": {
                    "mem": "ok%"
                },
                "ts": "0"
            }

            vllm4 = {
                "container_name": "vllm4",
                "uid": "52352352",
                "status": "running",
                "State": {
                    "Status": "running"
                },
                "gpu": {
                    "mem": "ok%"
                },
                "ts": "0"
            }

            vllm5 = {
                "container_name": "vllm5",
                "uid": "74545",
                "status": "running",
                "State": {
                    "Status": "running"
                },
                "gpu": {
                    "mem": "ok%"
                },
                "ts": "0"
            }
            res_data = [vllm1,vllm2,vllm3,vllm4,vllm5]
            return JSONResponse({"result_status": 200, "result_data": res_data})
        if req_data["method"] == "generate":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] generate >>>>>>>>>>>')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] generate >>>>>>>>>>> ')

            if req_data["vllmcontainer"] == "container_vllm_oai":
                VLLM_URL = f'http://{req_data["vllmcontainer"]}:{req_data["port"]}/v1/chat/completions'
                print(f'trying request vllm with da URL: {VLLM_URL}')
                try:
                    response = requests.post(VLLM_URL, json={
                        "model":req_data["model"],
                        "messages": [
                                        {
                                            "role": "user",
                                            "content": f'{req_data["prompt"]}'
                                        }
                        ]
                    })
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] response: {response}')
                    logging.info(f' [docker]  response: {response}') 
                    if response.status_code == 200:
                        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] status_code: {response.status_code}')
                        logging.info(f' [docker]  status_code: {response.status_code}') 
                        
                        
                        response_json = response.json()
                        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] response_json: {response_json}')
                        logging.info(f' [docker]  response_json: {response_json}') 
                        
                        
                        message_content = response_json["choices"][0]["message"]["content"]
                        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] message_content: {message_content}')
                        logging.info(f' [docker]  message_content: {message_content}') 
                        
                        return JSONResponse({"result_status": 200, "result_data": f'{message_content}'})              
                    else:
                        logging.info(f' [docker] response: {response}')
                        return JSONResponse({"result_status": 500, "result_data": f'ERRRR response.status_code {response.status_code} response{response}'})
                
                except Exception as e:
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                    return f'err {str(e)}'
                
                
            if req_data["vllmcontainer"] == "container_vllm_xoo": 
                VLLM_URL = f'http://{req_data["vllmcontainer"]}:{req_data["port"]}/vllm'
                print(f'trying request vllm with da URL: {VLLM_URL}')
                try:
                    response = requests.post(VLLM_URL, json={
                        "req_type":"generate",
                        "prompt":req_data["prompt"],
                        "temperature":float(req_data["temperature"]),
                        "top_p":float(req_data["top_p"]),
                        "max_tokens":int(req_data["max_tokens"])
                    })
                    if response.status_code == 200:
                        logging.info(f' [docker]  status_code: {response.status_code}') 
                        response_json = response.json()
                        logging.info(f' [docker]  response_json: {response_json}') 
                        response_json["result_data"] = response_json["result_data"]
                        return JSONResponse({"result_status": 200, "result_data": f'{response_json["result_data"]}'})
                    else:
                        logging.info(f' [docker] response: {response}')
                        return JSONResponse({"result_status": 500, "result_data": f'ERRRR response.status_code {response.status_code} response{response}'})
                
                except Exception as e:
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                    return JSONResponse({"result_status": 500, "result_data": f'err {str(e)}'})
            
            return JSONResponse({"result_status": 404, "result_data": f'{req_data["vllmcontainer"]} not found!'})
  
        if req_data["method"] == "logs":
            req_container = client.containers.get(req_data["model"])
            res_logs = req_container.logs()
            res_logs_str = res_logs.decode('utf-8')
            reversed_logs = "\n".join(res_logs_str.splitlines()[::-1])
            return JSONResponse({"result": 200, "result_data": reversed_logs})

        if req_data["method"] == "network":
            req_container = client.containers.get(req_data["container_name"])
            stats = req_container.stats(stream=False)
            return JSONResponse({"result": 200, "result_data": stats})

        if req_data["method"] == "list":
            res_container_list = client.containers.list(all=True)
            return JSONResponse([container.attrs for container in res_container_list])

        if req_data["method"] == "delete":
            req_container = client.containers.get(req_data["model"])
            req_container.stop()
            req_container.remove(force=True)
            return JSONResponse({"result": 200})

        if req_data["method"] == "stop":
            req_container = client.containers.get(req_data["model"])
            req_container.stop()
            return JSONResponse({"result": 200})

        if req_data["method"] == "start":
            req_container = client.containers.get(req_data["model"])
            req_container.start()
            return JSONResponse({"result": 200})

        if req_data["method"] == "load":
            print(f' * ! * ! * trying to load ....  0 ')
            # VLLM_URL = f'http://container_vllm_xoo:{os.getenv("VLLM_PORT")}/vllm'
            # if req_data["vllmcontainer"] == "container_vllm_xoo":  ....
            
            print(f'  * ! * ! *  calling stop_vllm_container()')
            res_stop_vllm_container = await stop_vllm_container()
            print(f'  * ! * ! *  calling stop_vllm_container() -> res_stop_vllm_container -> {res_stop_vllm_container}')      
            
            # check if container exists with this model if yes start ..
            if req_data["vllmcontainer"] == "container_vllm_oai":
                return JSONResponse({"result_status": 500, "result_data": f'vllm/vllm-openai:latest load not supported!'})
            
            if req_data["vllmcontainer"] == "container_vllm_xoo":                
                print(f'  * ! * ! *  starting container_vllm_xoo ...')
                req_container = client.containers.get(req_data["vllmcontainer"])
                print(f'  * ! * ! *  is started? [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] ...')
                req_container.start()
                print(f'  * ! * ! *  is started? [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] ... zzz 60 sec safety')
                time.sleep(60)
                print(f'  * ! * ! *  is started? [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] ... zzz .. awake! loading ...')
                VLLM_URL = f'http://{req_data["vllmcontainer"]}:{req_data["port"]}/vllm'
                print(f' * ! * ! * trying to load ....  1 VLLM_URL {VLLM_URL}')
                try:
                    response = requests.post(VLLM_URL, json={
                        "req_type":"load",
                        "max_model_len":int(req_data["max_model_len"]),
                        "tensor_parallel_size":int(req_data["tensor_parallel_size"]),
                        "gpu_memory_utilization":float(req_data["gpu_memory_utilization"]),
                        "model":str(req_data["model"])
                    })
                    print(f' * ! * ! * trying to load ....  3 response {response}')
                    if response.status_code == 200:
                        print(f' * ! * ! * trying to load ....  4 status_code: {response.status_code}')
                        
                        response_json = response.json()
                        print(f' * ! * ! * trying to load ....  5 response_json: {response_json}')
                        print(f' * ! * ! * trying to load ....  6 response_json["result_data"]: {response_json["result_data"]}')
                        return JSONResponse({"result_status": 200, "result_data": f'{response_json["result_data"]}'})
                    else:
                        print(f' * ! * ! * trying to load .... 7 ERRRRR')
                        return JSONResponse({"result_status": 500, "result_data": f'ERRRRRR'})
                
                except Exception as e:
                        print(f' * ! * ! * trying to load .... 8 ERRRRR')
                        return JSONResponse({"result_status": 500, "result_data": f'ERRRRRR 8'})

            return JSONResponse({"result_status": 500, "result_data": f'{req_data["vllmcontainer"]} load not supported!'})
        
        if req_data["method"] == "create":
            try:
                    
                                
                print(f'  !!!!!  calling stop_vllm_container()')
                res_stop_vllm_container = await stop_vllm_container()
                print(f'  !!!!!  calling stop_vllm_container() -> res_stop_vllm_container -> {res_stop_vllm_container}')      
                
            
                req_container_name = str(req_data["model"]).replace('/', '_')
                req_container_name = req_container_name.split('_')[0]
                ts = str(int(datetime.now().timestamp()))
                req_container_name = f'container_vllm_{req_container_name}_{ts}'
                
                # req_container_name = f'container_vllm_asdf'
                
                print(f' !!!!! calling req_container_name: {req_container_name}')
                
                if req_data["image"] == "vllm/vllm-openai:latest":
                    print(f' !!!!! create found "vllm/vllm-openai:latest" !')
                
                if "xoo4foo/" in req_data["image"]:
                    print(f' !!!!! create found "xoo4foo/" !')
                
                

                print(f' ************ calling stop_vllm_container()')
                res_stop_vllm_container = await stop_vllm_container()
                print(f' ************ calling stop_vllm_container() -> res_stop_vllm_container -> {res_stop_vllm_container}')      
                
                if req_data["image"] == "vllm/vllm-openai:latest":
                    print(f' !!!!! create found "vllm/vllm-openai:latest" !')
                    res_container = client.containers.run(
                        build={"context": f'./{req_container_name}'},
                        image=req_data["image"],
                        runtime=req_data["runtime"],
                        ports={
                            f'{req_data["port"]}/tcp': ("0.0.0.0", req_data["port"])
                        },
                        container_name=f'{req_container_name}',
                        volumes={
                            "/logs": {"bind": "/logs", "mode": "rw"},
                            "/home/cloud/.cache/huggingface": {"bind": "/root/.cache/huggingface", "mode": "rw"},
                            "/models": {"bind": "/root/.cache/huggingface/hub", "mode": "rw"}
                        },
                        shm_size=f'{req_data["shm_size"]}',
                        network=network_name,
                        environment={
                            "NCCL_DEBUG": "INFO"
                        },
                        command=[
                            f'--model {req_data["model"]}',
                            f'--port {req_data["port"]}',
                            f'--tensor-parallel-size {req_data["tensor_parallel_size"]}',
                            f'--gpu-memory-utilization {req_data["gpu_memory_utilization"]}',
                            f'--max-model-len {req_data["max_model_len"]}'
                        ]
                    )
                    container_id = res_container.id
                    return JSONResponse({"result_status": 200, "result_data": str(container_id)})
                
                if "xoo4foo/" in req_data["image"]:
                    print(f' !!!!! create found "xoo4foo/" !')
                    print(f' !!!!! using req_container_name: {req_container_name} !')

                    res_container = client.containers.run(
                        image=req_data["image"],
                        name=req_container_name,
                        runtime=req_data["runtime"],
                        shm_size=req_data["shm_size"],
                        network=network_name,
                        detach=True,
                        environment={
                            'NCCL_DEBUG': 'INFO',
                            'VLLM_PORT': req_data["port"]
                        },
                        device_requests=[
                            docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                        ],
                        ports={f'{req_data["port"]}': req_data["port"]},
                        volumes={
                            '/logs': {'bind': '/logs', 'mode': 'rw'},
                            '/models': {'bind': '/models', 'mode': 'rw'}
                        },
                        command=[
                            "python", "app.py",
                            "--model", req_data["model"],
                            "--port", str(req_data["port"]),
                            "--tensor-parallel-size", str(req_data["tensor_parallel_size"]),
                            "--gpu-memory-utilization", str(req_data["gpu_memory_utilization"]),
                            "--max-model-len", str(req_data["max_model_len"])
                        ]
                    )
                    

                    container_id = res_container.id
                    return JSONResponse({"result_status": 200, "result_data": str(container_id)})
                        
            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return JSONResponse({"result_status": 500, "result_data": f'{e}'})

    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": f'{e}'})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=f'{os.getenv("BACKEND_IP")}', port=int(os.getenv("BACKEND_PORT")))