import os
import threading
#使用的库
from pathlib import Path
import subprocess
import pandas as pd
import shutil
import os
import time
import re
import gc
import requests
import zipfile
import threading
import time
import socket
from concurrent.futures import ProcessPoolExecutor
import time
# import wandb

# import wandb
os.system("pip install nvidia-ml-py3")
os.chdir(f"/home/xlab-app-center")
if os.path.isdir("/home/xlab-app-center/stable-diffusion-webui"):
    os.chdir(f"/home/xlab-app-center/stable-diffusion-webui")
    os.system("git pull")
else:
    os.system(f"git clone https://openi.pcl.ac.cn/lgkkey/sd-webui.git /home/xlab-app-center/stable-diffusion-webui")
os.chdir(f"/home/xlab-app-center")
os.system(f"cp /home/xlab-app-center/styles.csv /home/xlab-app-center/stable-diffusion-webui/styles.csv")
os.chdir(f"/home/xlab-app-center/stable-diffusion-webui")
os.system(f"git lfs install")
os.system(f"git reset --hard")
os.chdir(f"/home/xlab-app-center/stable-diffusion-webui/extensions")

plugins = [
    "https://openi.pcl.ac.cn/2575044704/stable-diffusion-webui-localization-zh_CN2",
    "https://gitcode.net/ranting8323/multidiffusion-upscaler-for-automatic1111",
    "https://gitcode.net/ranting8323/adetailer",
    "https://gitcode.net/ranting8323/sd-webui-prompt-all-in-one",
    "https://github.com/Uminosachi/sd-webui-inpaint-anything.git",
    "https://gitcode.net/ranting8323/a1111-sd-webui-tagcomplete",
    "https://gitcode.net/nightaway/sd-webui-infinite-image-browsing",
    "https://openi.pcl.ac.cn/2575044704/sd-extension-system-info",
    "https://openi.pcl.ac.cn/2575044704/batchlinks-webui"
]

# https://hf-mirror.com/marcy1111/majicmixRealistic_v7/resolve/main/majicmixRealistic_v7.safetensors
for plugin in plugins:
    os.system(f"git clone {plugin}")
os.makedirs('/home/xlab-app-center/stable-diffusion-webui/models/adetailer', exist_ok=True)
os.chdir(f"/home/xlab-app-center/stable-diffusion-webui/models/adetailer")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://hf-mirror.com/Bingsu/adetailer/resolve/main/hand_yolov8s.pt -d /home/xlab-app-center/stable-diffusion-webui/models/adetailer -o hand_yolov8s.pt")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://hf-mirror.com/Bingsu/adetailer/resolve/main/hand_yolov8n.pt -d /home/xlab-app-center/stable-diffusion-webui/models/adetailer -o hand_yolov8n.pt")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://hf-mirror.com/datasets/ACCC1380/private-model/resolve/main/kaggle/input/museum/131-half.safetensors -d /home/xlab-app-center/stable-diffusion-webui/models/Stable-diffusion -o 131-half.safetensors")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://hf-mirror.com/datasets/ACCC1380/private-model/resolve/main/ba.safetensors -d /home/xlab-app-center/stable-diffusion-webui/models/Lora -o ba.safetensors")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://hf-mirror.com/datasets/ACCC1380/private-model/resolve/main/racaco2.safetensors -d /home/xlab-app-center/stable-diffusion-webui/models/Lora -o racaco2.safetensors")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://hf-mirror.com/coinz/Add-detail/resolve/main/add_detail.safetensors -d /home/xlab-app-center/stable-diffusion-webui/models/Lora -o add_detail.safetensors")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://hf-mirror.com/datasets/VASVASVAS/vae/resolve/main/pastel-waifu-diffusion.vae.pt -d /home/xlab-app-center/stable-diffusion-webui/models/VAE -o pastel-waifu-diffusion.vae.pt")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://download.openxlab.org.cn/models/camenduru/sdxl-refiner-1.0/weight//sd_xl_refiner_1.0.safetensors -d /home/xlab-app-center/stable-diffusion-webui/models/Stable-diffusion -o sd_xl_refiner_1.0.safetensors")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://download.openxlab.org.cn/models/camenduru/cyber-realistic/weight//cyberrealistic_v32.safetensors -d /home/xlab-app-center/stable-diffusion-webui/models/Stable-diffusion -o cyberrealistic_v32.safetensors")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://hf-mirror.com/marcy1111/majicmixRealistic_v7/resolve/main/majicmixRealistic_v7.safetensors -d /home/xlab-app-center/stable-diffusion-webui/models/Stable-diffusion -o majicmixRealistic_v7.safetensors")

os.chdir(f"/home/xlab-app-center/stable-diffusion-webui")
print('webui launching...')
package_envs = [
    {"env": "STABLE_DIFFUSION_REPO", "url": os.environ.get('STABLE_DIFFUSION_REPO', "https://gitcode.net/overbill1683/stablediffusion")},
    {"env": "STABLE_DIFFUSION_XL_REPO", "url": os.environ.get('STABLE_DIFFUSION_XL_REPO', "https://gitcode.net/overbill1683/generative-models")},
    {"env": "K_DIFFUSION_REPO", "url": os.environ.get('K_DIFFUSION_REPO', "https://gitcode.net/overbill1683/k-diffusion")},
    {"env": "CODEFORMER_REPO", "url": os.environ.get('CODEFORMER_REPO', "https://gitcode.net/overbill1683/CodeFormer")},
    {"env": "BLIP_REPO", "url": os.environ.get('BLIP_REPO', "https://gitcode.net/overbill1683/BLIP")},
]
os.environ["PIP_INDEX_URL"] = "https://mirrors.aliyun.com/pypi/simple/"
for i in package_envs:
    os.environ[i["env"]] = i["url"]

# WandB登录
# os.system('wandb login 5c00964de1bb95ec1ab24869d4c523c59e0fb8e3')

# 初始化WandB项目
# wandb.init(project="gpu-temperature-monitor")

show_shell_info = False
def run(command, cwd=None, desc=None, errdesc=None, custom_env=None,try_error:bool=True) -> str:
    global show_shell_info
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "cwd": cwd,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not show_shell_info:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        if try_error:
            print((RuntimeError("\n".join(error_bits))))
        else:
            raise RuntimeError("\n".join(error_bits))

    if show_shell_info:
        print((result.stdout or ""))
    return (result.stdout or "")

def mkdirs(path, exist_ok=True):
    if path and not Path(path).exists():
        os.makedirs(path,exist_ok=exist_ok)
proxy_path={
    '/sd2/':'http://127.0.0.1:7862/',
    '/sd3/':'http://127.0.0.1:7863/'
} # 增加一个comfyui的代理
server_port=7860 # webui 默认端口
_server_port = locals().get('server_port') or globals().get('server_port') or 7860


_proxy_path = locals().get('proxy_path') or globals().get('proxy_path') or {}
# nginx 反向代理配置文件
def echoToFile(content:str,path:str):
    if path.find('/') >= 0:
        _path = '/'.join(path.split('/')[:-1])
        run(f'''mkdir -p {_path}''')
    with open(path,'w') as sh:
        sh.write(content)
# 检查网络
def check_service(host, port):
    try:
        socket.create_connection((host, port), timeout=5)
        return True
    except socket.error:
        return False
def localProxy():
    os.system('sudo apt install nginx -y')
    
    _proxy_path['/'] = f'http://127.0.0.1:{_server_port+1}/'
    _proxy_path['/1/'] = f'http://127.0.0.1:{_server_port+2}/'
    
    def getProxyLocation(subPath:str, localServer:str):
        return '''
    location '''+ subPath +'''
    {
        proxy_pass '''+ localServer +''';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header REMOTE-HOST $remote_addr;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection upgrade;
        proxy_http_version 1.1;
        proxy_connect_timeout 10m;
        proxy_read_timeout 10m;
    }
    
    '''
    
    conf = '''
server
{
    listen '''+str(_server_port)+''';
    listen [::]:'''+str(_server_port)+''';
    server_name 127.0.0.1 localhost 0.0.0.0 "";
    
    if ($request_method = OPTIONS) {
        return 200;
    }
    fastcgi_send_timeout                 10m;
    fastcgi_read_timeout                 10m;
    fastcgi_connect_timeout              10m;
    
    '''+ ''.join([getProxyLocation(key,_proxy_path[key]) for key in _proxy_path.keys()]) +'''
}
'''
    echoToFile(conf,'/home/xlab-app-center/etc/nginx/conf.d/proxy_nginx.conf')
    if not check_service('localhost',_server_port):
        run(f'''nginx -c /home/xlab-app-center/etc/nginx/nginx.conf''')
    run(f'''nginx -s reload''')
    

def start():
    #try:
    #    print('启动proxy')
    #    threading.Thread(target = localProxy,daemon=True).start()
    #except Exception as e:
    #    # 在这里处理异常的代码
    #    print(f"proxy An error occurred: {e}")
    try:
    #安装环境
        os.system(f"python launch.py --api --xformers --exit --enable-insecure-extension-access --gradio-queue --disable-safe-unpickle")
        #time.sleep(5)
        
        command = "python launch.py --api --xformers --ui-settings-file /home/xlab-app-center/config.json --ui-config-file /home/xlab-app-center/ui-config.json --gradio-queue --disable-safe-unpickle"

        
        process = subprocess.Popen(command, shell=True)
        time.sleep(120)
        # os.system(f"{command} --port=7861 --ngrok=2KPyfzQrHit97J02tARy1ckHJYd_69rJbgjpjnVVeuXD3j9tv ")
        os.system(f"{command} --port=7861")
    except Exception as e:
        # 在这里处理异常的代码
        print(f"启动SD发生错误: {e}")
# Create threads for each function
# wandb_thread = threading.Thread(target=monitor_gpu)
start_thread = threading.Thread(target=start)

# Start the threads
# wandb_thread.start()
start_thread.start()

# Wait for both threads to finish
# wandb_thread.join()
start_thread.join()

while True:
    time.sleep(10000)
