# set random seed
# generate 2k dataset w/ stable diff
# use category names for imagenet with style prompts
# gpt-api for llm style prompts
# local docker container w/ replicate server for stable diff 

"""
pip install replicate

https://replicate.com/stability-ai/sdxl
docker run --name sdxl r8.im/stability-ai/sdxl@sha256:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b
docker commit sdxl sdxl
docker run -p 5000:5000 --gpus=all sdxl
"input": {
    "width": 768,
    "height": 768,
    "prompt": "An astronaut riding a rainbow unicorn, cinematic, dramatic",
    "refine": "expert_ensemble_refiner",
    "scheduler": "K_EULER",
    "lora_scale": 0.6,
    "num_outputs": 1,
    "guidance_scale": 7.5,
    "apply_watermark": false,
    "high_noise_frac": 0.8,
    "negative_prompt": "",
    "prompt_strength": 0.8,
    "num_inference_steps": 25
}
"""

import base64
import os
import subprocess
import uuid
import time
from io import BytesIO
from typing import List

import requests
from PIL import Image


def nuke_docker():
    containers = os.popen("docker ps -aq").read().strip()
    if containers:
        os.system(f"docker kill {containers}")
        os.system(f"docker stop {containers}")
        os.system(f"docker rm {containers}")
    os.system("docker container prune -f")


def make_docker(name: str):
    nuke_docker()
    docker_process = subprocess.Popen(
        ["docker", "run", "--rm", "-p", "5000:5000", "--gpus=all", name]
    )
    time.sleep(20)  # Let the docker container startup
    return docker_process

def gen(cats: List[str], styles: List[str], base_output_dir: str = "/home/oop/dev/data/"):
    session_id = str(uuid.uuid4())[:6]
    output_dir = os.path.join(base_output_dir, f"sdxlimgnet.{session_id}")
    os.makedirs(output_dir, exist_ok=True)
    # docker_process = make_docker("sdxl")
    for cat in cats:
        cat_dir = os.path.join(output_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)
        for style in styles:
            response = requests.post(
                "http://localhost:5000/predictions",
                headers={"Content-Type": "application/json"},
                json={
                    "input": {
                        "width": 768,
                        "height": 768,
                        "prompt": f"{cat} in the style of {style}",
                        "refine": "expert_ensemble_refiner",
                        "scheduler": "K_EULER",
                        "lora_scale": 0.6,
                        "num_outputs": 4,
                        "guidance_scale": 7.5,
                        "apply_watermark": False,
                        "high_noise_frac": 0.8,
                        "negative_prompt": "",
                        "prompt_strength": 1.0, # 0.8,
                        "num_inference_steps": 8, # 25,
                        "disable_safety_checker": True,
                    }
                },
            )
            print(response.json())
            for j in range(4):
                img = Image.open(BytesIO(base64.b64decode(response.json()["output"][j].split(",")[1])))
                img = img.resize((224, 224))
                img.save(os.path.join(cat_dir, f"{style}.{j}.jpg"))
    # docker_process.terminate()
    # nuke_docker()


if __name__ == "__main__":
    from imagenet_cats import IMAGENET_CLASSES
    styles = [
        "realistic photograph",
        "cartoon image",
        "high definition 3d render",
        "grainy early internet image",
    ]
    gen(list(IMAGENET_CLASSES.values()), styles)