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
from multiprocessing import Process
import time
try:
    from git.repo import Repo
    from git.repo.fun import is_git_dir
except ImportError as e:
    os.system("pip install gitpython")
    from git.repo import Repo
    from git.repo.fun import is_git_dir

# import wandb
install_path = '/home/xlab-app-center'
rename_repo = 'stable-diffusion-webui'
download_tool = 'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M'
package_envs = [
    {"env": "STABLE_DIFFUSION_REPO", "url": os.environ.get('STABLE_DIFFUSION_REPO', "https://github.com/Stability-AI/stablediffusion.git")},
    {"env": "STABLE_DIFFUSION_XL_REPO", "url": os.environ.get('STABLE_DIFFUSION_XL_REPO', "https://github.com/Stability-AI/generative-models.git")},
    {"env": "K_DIFFUSION_REPO", "url": os.environ.get('K_DIFFUSION_REPO', "https://github.com/crowsonkb/k-diffusion.git")},
    {"env": "CODEFORMER_REPO", "url": os.environ.get('CODEFORMER_REPO', "https://github.com/sczhou/CodeFormer.git")},
    {"env": "BLIP_REPO", "url": os.environ.get('BLIP_REPO', "https://github.com/salesforce/BLIP.git")},
]
os.environ["PIP_INDEX_URL"] = "https://mirrors.aliyun.com/pypi/simple/"
for i in package_envs:
    os.environ[i["env"]] = i["url"]

def gitUtils(repo_url,local_path,branch='',commit_hash="",tag="",recursive=False,update_last=False,check_commit=True):
    print("repo_url:",repo_url,",clone path:",local_path,"==>start clone...")
    try:
        repo=None
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        git_local_path = os.path.join(local_path, '.git')   
        if not is_git_dir(git_local_path):
            if branch=="":
                repo = Repo.clone_from(repo_url, to_path=local_path,recursive=recursive)
            else:
                repo = Repo.clone_from(repo_url, to_path=local_path, branch=branch,recursive=recursive)
        else:
            repo = Repo(local_path)
        if not  update_last:    
            if branch=="":
                branch=repo.active_branch.name
            else:
                if (repo.active_branch.name!=branch):
                    repo.git.checkout(branch)
            if(commit_hash!=''):
                try:
                    if (check_commit and repo.active_branch.commit.hexsha!=commit_hash):
                        repo.git.reset('--hard', commit_hash)
                    else:
                        repo.git.reset('--hard', commit_hash)
                except Exception as e:
                    print(f"commit_hash:{commit_hash} not found,use default branch:{branch}")
            elif(tag!=''):
                find_it=""
                for tag_ in repo.tags:
                    if tag_.name==tag:
                        find_it=tag
                        if (check_commit and tag_.commit.hexsha!=repo.active_branch.commit.hexsha):
                            repo.git.checkout(tag_.name)
                            break
                        else:
                            repo.git.checkout(tag_.name)
                            break
                
                tag=find_it
        else:
            repo.head.reset(commit=repo.active_branch.commit, index=True, working_tree=True)
            repo.git.pull()
            
        if (len(repo.submodules)>0):
            repo.submodule_update(init=True,recursive=True)


        print(f"finish clone: {repo_url} ,save to: {local_path}")
        print(f"curr info: branch:{repo.active_branch.name}")
        print(f"         : tag:{tag}")
        print(f"         : submodule:{len(repo.submodules)}")
        print(f"         : commit_hash:{repo.head.commit.hexsha}")   
    except Exception as e: 
        print("down error:\n",e)
        print("try to use git cmd")
        os.system(f"git clone {repo_url} {local_path}")
         
def model_download(models, type_w):
    for model in models:
        try:
            download_files(model, type_w)
        except :
            print(f'{model} 下载失败')

def download_files(url, source):
    curr_path=os.getcwd()
    if '@' in url and (not url.startswith('http://') and not url.startswith('https://')):
        parts = url.split('@', 1)
        name = parts[0]
        url = parts[1]
        rename = f"-o '{name}'"
        if 'huggingface.co' in url:
            url = url.replace("huggingface.co", "hf-mirror.com")
    else:
        if ('huggingface.co' or 'hf-mirror.com' or 'huggingface.sukaka.top') in url:
            url = url.replace("huggingface.co", "hf-mirror.com")
            match_name = re.search(r'/([^/?]+)(?:\?download=true)?$', url).group(1)
            if match_name:
                rename = f"-o '{match_name}'"
            else:
                rename = ''
        else:
            rename = ''
    source_dir = f'{install_path}/{rename_repo}/{source}'
    os.makedirs(source_dir, exist_ok=True)
    os.chdir(source_dir)
    os.system(f"{download_tool} '{url}' {rename}")
    os.chdir(curr_path)
    

def download_files_wget(url, source):
    wget_tool="wget -c -o "
    curr_path=os.getcwd()
    if '@' in url and (not url.startswith('http://') and not url.startswith('https://')):
        parts = url.split('@', 1)
        name = parts[0]
        url = parts[1]
        rename = name

    source_dir = f'{install_path}/{rename_repo}/{source}'
    os.makedirs(source_dir, exist_ok=True)
    os.chdir(source_dir)
    print(f"do: {wget_tool} {rename} '{url}' ")
    os.system(f"{wget_tool} {rename} '{url}'")
    print(f"{wget_tool} {rename} '{url}' finish")
    os.chdir(curr_path)

def download_extensions(extensions):
    git_local_path = os.path.join(install_path, rename_repo, 'extensions')
    os.chdir(git_local_path)
    for extension in extensions:
        # os.system(f'git clone {extension}')
        try:
            gitUtils(extension.get("repo_url"),os.path.join(git_local_path,extension.get("save_name")),
                    branch=extension.get("branch",""),
                    commit_hash=extension.get("commit_hash",""),
                    tag=extension.get("tag",""),
                    recursive=extension.get("recursive",False),
                    update_last=extension.get("update_last",False),
                    check_commit=extension.get("check_commit",True)
                    )
        except Exception as e:
            print(f"{extension.get('repo_url')} clone error:\n",e)


os.system("pip install nvidia-ml-py3")
os.chdir(f"/home/xlab-app-center")
gitUtils(f"https://openi.pcl.ac.cn/lgkkey/sd-webui.git",f"/home/xlab-app-center/stable-diffusion-webui",recursive=True,update_last=True)
os.system(f"cp /home/xlab-app-center/styles.csv /home/xlab-app-center/stable-diffusion-webui/styles.csv")
os.chdir(f"/home/xlab-app-center/stable-diffusion-webui")
os.system(f"git lfs install")
os.system(f"git reset --hard")


plugins = [
    {"repo_url":"https://openi.pcl.ac.cn/2575044704/stable-diffusion-webui-localization-zh_CN2","save_name":"stable-diffusion-webui-localization-zh_CN2","branch":"","commit_hash":"","tag":""},
    {"repo_url":"https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111.git","save_name":"multidiffusion-upscaler-for-automatic1111","branch":"","commit_hash":"574a0963133a34815f65bfaf985c19de54fdf323","tag":""},
    {"repo_url":"https://github.com/Bing-su/adetailer.git","save_name":"adetailer","branch":"","commit_hash":"a7d961131e879ea8a930034a21a2dee21b173e8c","tag":""},
    {"repo_url":"https://github.com/Physton/sd-webui-prompt-all-in-one.git","save_name":"sd-webui-prompt-all-in-one","branch":"","commit_hash":"d69645e6a6701c5117e0a874d1ef80d5cb5d55cc","tag":""},
    {"repo_url":"https://github.com/Uminosachi/sd-webui-inpaint-anything.git","save_name":"sd-webui-inpaint-anything","branch":"","commit_hash":"ae6cc075f6c0f0dc360f2998181f4f8cbebd3622","tag":""},
    {"repo_url":"https://github.com/DominikDoom/a1111-sd-webui-tagcomplete.git","save_name":"a1111-sd-webui-tagcomplete","branch":"","commit_hash":"3ef2a7d206d95eefec4cc7950c8d3b940dc18f9f","tag":""},
    {"repo_url":"https://github.com/zanllp/sd-webui-infinite-image-browsing.git","save_name":"sd-webui-infinite-image-browsing","branch":"","commit_hash":"83c8845e345bb9b0ec19f0e4a6164fea50fc3710","tag":""},
    {"repo_url":"https://github.com/vladmandic/sd-extension-system-info.git","save_name":"sd-extension-system-info","branch":"","commit_hash":"c88e83d403e1cae478df870fa2dd277d2028dc34","tag":""},
    # {"repo_url":"https://openi.pcl.ac.cn/2575044704/batchlinks-webui","save_name":"batchlinks-webui","branch":"","commit_hash":"","tag":""},
    {"repo_url":"https://github.com/Mikubill/sd-webui-controlnet.git","save_name":"sd-webui-controlnet","branch":"","commit_hash":"8bbbd0e55ef6e5d71b09c2de2727b36e7bc825b0","tag":""},
]
# 'https://github.com/Mikubill/sd-webui-controlnet.git' 
# https://hf-mirror.com/marcy1111/majicmixRealistic_v7/resolve/main/majicmixRealistic_v7.safetensors

download_extensions(plugins)


def down_adetailer_model():
    print("down adteiler model")
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
    print("finish adetailer model thread!!")

Process(target=down_adetailer_model).start()

# other model
sd_models = [
   "Anything_XL@https://civitai.com/api/download/models/384264?type=Model&format=SafeTensor&size=full&fp=fp16"
]
lora_models = [
     "Labiaplasty_v2@https://civitai.com/api/download/models/182404?type=Model&format=SafeTensor"
     
]

vae_models = []


controlnet_models = [
'https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors',
'https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_shuffle_fp16.safetensors',
'https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors',
'https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors',
'https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors',
'https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors',
'https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors',
'https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors',
'https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors',
'https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_scribble_fp16.safetensors',
'https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors',
'https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors',
'https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11u_sd15_tile_fp16.safetensors',
'https://huggingface.co/DionTimmer/controlnet_qrcode-control_v1p_sd15/resolve/main/control_v1p_sd15_qrcode.safetensors',
]
embedding_models = [
]
hypernetwork_models = []

esrgan_models = []

if os.path.isfile("/home/xlab-app-center/stable-diffusion-webui/extensions/sd-webui-controlnet/requirements.txt"):
    os.chdir('/home/xlab-app-center/stable-diffusion-webui/extensions/sd-webui-controlnet')
    os.system("python -m pip install -r requirements.txt")
    os.system("python install.py")
    p1=Process(target=model_download,args=(controlnet_models,'extensions/sd-webui-controlnet/models'))
    p1.start()
# model_download(controlnet_models, 'extensions/sd-webui-controlnet/models')
download_files_wget(sd_models[0], 'models/Stable-diffusion')
download_files_wget(lora_models[0], 'models/Lora')
model_download(vae_models, 'models/VAE')
model_download(hypernetwork_models, 'models/hypernetworks')
model_download(embedding_models, 'embeddings')
model_download(esrgan_models, 'models/ESRGAN')


os.chdir(f"/home/xlab-app-center/stable-diffusion-webui")
print('webui launching...')


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
