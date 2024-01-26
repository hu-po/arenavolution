import argparse
import base64
import os
import requests
import random
import shutil
import subprocess
import time
import uuid
import yaml

from io import BytesIO
from PIL import Image
from openai import OpenAI


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_players", type=int, default=6)
parser.add_argument("--num_rounds", type=int, default=2)
parser.add_argument("--base_dir", type=str, default="/home/oop/dev/data/")
parser.add_argument("--data_dir", type=str, default=None)
# parser.add_argument("--data_dir", type=str, default="/home/oop/dev/data/centipede_chickadee")
parser.add_argument("--num_categories", type=int, default=4)
parser.add_argument("--dataset_size", type=int, default=32)
parser.add_argument("--dataset_split", type=float, default=0.8)
args = parser.parse_args()

print("🧙‍♂️ Starting Einvolution")
random.seed(args.seed)
session_id = str(uuid.uuid4())[:6]
base_dir = os.path.join(args.base_dir, f"einvol.{session_id}")
os.makedirs(base_dir, exist_ok=True)
print(f"base directory at {base_dir}")
player_dir = os.path.join(base_dir, "players")
os.makedirs(player_dir, exist_ok=True)
print(f"player directory at {player_dir}")
logs_dir = os.path.join(base_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)
print(f"logs directory at {logs_dir}")
ckpt_dir = os.path.join(base_dir, "ckpt")
os.makedirs(ckpt_dir, exist_ok=True)
print(f"ckpt directory at {ckpt_dir}")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if args.data_dir is None:
    dataset_id = str(uuid.uuid4())[:6]
    print(f"No data directory specified, generating new dataset {dataset_id}")
    data_dir = os.path.join(base_dir, f"data.{dataset_id}")
    os.makedirs(data_dir, exist_ok=True)
    print(f"data directory at {data_dir}")
    train_dir = os.path.join(data_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    print(f"train directory at {train_dir}")
    test_dir = os.path.join(data_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    print(f"test directory at {test_dir}")
    # Check if Docker is already running
    docker_ps_process = subprocess.Popen(["docker", "ps"], stdout=subprocess.PIPE)
    docker_ps_output, _ = docker_ps_process.communicate()
    if "sdxl" in docker_ps_output.decode():
        print("Docker is already running.")
        sdxl_docker_proc = None
    else:
        os.system("docker kill $(docker ps -aq) && docker rm $(docker ps -aq)")
        sdxl_docker_proc = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "-p",
                "5000:5000",
                "--gpus=all",
                "-v /home/oop/dev/data/sdxl/sdxl-cache:/path/inside/container/sdxl-cache",
                "-v /home/oop/dev/data/sdxl/safety-cache:/path/inside/container/safety-cache",
                "sdxl",
            ],
        )
        time.sleep(20)  # Let the docker container startup
    # use gpt to generate Nc categories and Ns Styles
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""
You are a sampling machine that provides perfectly sampled words.
You provide samples from the distribution of semantic visual concepts.
These words will be used as categories for an image classification task.
Return a comma separated list of {args.num_categories} words with no spaces.
        """,
            }
        ],
        model="gpt-4-1106-preview",
        temperature=1.7,
        max_tokens=6 * args.num_categories,
    )
    reply: str = response.choices[0].message.content
    categories = reply.split(",")
    print(f"Categories: {categories}")
    assert len(categories) == args.num_categories
    num_examples_per_category = args.dataset_size // args.num_categories
    for i, cat in enumerate(categories):
        print(f"Generating images for category {cat}")
        os.makedirs(os.path.join(train_dir, cat), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cat), exist_ok=True)
        for j in range(num_examples_per_category // 4):  # SDXL does 4 images at a time
            negative_prompt = categories[(i + 1) % len(categories)]
            response = requests.post(
                "http://localhost:5000/predictions",
                headers={"Content-Type": "application/json"},
                json={
                    "input": {
                        "width": 768,
                        "height": 768,
                        "prompt": cat,
                        "refine": "expert_ensemble_refiner",
                        "scheduler": "K_EULER",
                        "lora_scale": 0.6,
                        "num_outputs": 4,
                        "guidance_scale": 7.5,
                        "apply_watermark": False,
                        "high_noise_frac": 0.8,
                        "negative_prompt": negative_prompt,
                        "prompt_strength": 1.0,  # 0.8,
                        "num_inference_steps": 8,  # 25,
                        "disable_safety_checker": True,
                    }
                },
            )
            for k in range(4):
                img_id = str(uuid.uuid4())[:6]
                img = Image.open(
                    BytesIO(
                        base64.b64decode(response.json()["output"][k].split(",")[1])
                    )
                )
                img = img.resize((224, 224))
                if j < args.dataset_split * num_examples_per_category:
                    img.save(os.path.join(train_dir, cat, f"{img_id}.png"))
                else:
                    img.save(os.path.join(test_dir, cat, f"{img_id}.png"))
    if sdxl_docker_proc is not None:
        sdxl_docker_proc.terminate()
        os.system("docker kill $(docker ps -aq) && docker rm $(docker ps -aq)")

# Seed with the players in the local directory "players"
seed_players_dir = os.path.join(os.getcwd(), "players")
players = os.listdir(seed_players_dir)
for player in players:
    shutil.copy(os.path.join(seed_players_dir, player), player_dir)

for round in range(args.num_rounds):
    print(f"Starting evolution rounds {round}")
    # reproduce to fill in missing players
    while len(players) < args.num_players:
        child_id = str(uuid.uuid4())[:6]
        print(f"Creating child {child_id}")
        parents = random.sample(players, 2)
        print(f"Reproducing {parents[0]} and {parents[1]}")
        child_prompt = """
    You are a machine learning expert coder.
    You excel at creating novel model architectures.
    You use PyTorch and make use of the einops library.
    You will be given several blocks of code.
    Create a new block of code inspired by the given blocks.
    The block of code should be called `Block` and should be a subclass of `nn.Module`.
    Make sure the kwarg `num_classes` is present in the `__init__` method.
    """
        for parent in parents:
            parent_filepath = os.path.join(player_dir, parent)
            with open(parent_filepath, "r") as f:
                child_prompt += f"<block>{f.read()}</block>"
        child_prompt += "Reply only with valid code. Do not explain."
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": child_prompt}],
            model="gpt-4-1106-preview",
            temperature=0.9,
            max_tokens=512,
        )
        # TODO: test the child code before saving
        child_filename = f"child.{child_id}.py"
        child_filepath = os.path.join(player_dir, child_filename)
        with open(child_filepath, "w") as f:
            f.write(response.choices[0].message.content)
        players.append(child_filename)

    best_scores = {}
    results_filepath = os.path.join(base_dir, "results.yaml")
    for player in players:
        print(f"Running traineval for {player}")
        model_filepath = os.path.join(player_dir, player)
        os.system("docker kill $(docker ps -aq) && docker rm $(docker ps -aq)")
        player_docker_proc = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "--gpus 0",
                f"-v {model_filepath}:/src/model.py",
                f"-v {ckpt_dir}:/ckpt",
                f"-v {logs_dir}:/logs",
                f"-v {data_dir}:/data",
                "evolver",
                f"--child_name={player}",
            ]
        )
        player_docker_proc.wait()
        if player_docker_proc.returncode != 0:
            print(f"Error occurred when training player {player}")
            best_scores[player] = 0.0
        else:
            print(f"Trained player {player}")
            with open(results_filepath, "r") as f:
                player_results = yaml.safe_load(f)
            best_scores[player] = player_results[player]["test_accuracy"]
        print(f"Player {player} result {best_scores[player]}")

    # Remove the lowest 50% of the players
    sorted_players = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"Sorted players: {sorted_players}")
    top_players = [x[0] for x in sorted_players[: len(sorted_players) // 2]]
    print(f"Top players: {top_players}")
    for player in players:
        if player not in top_players:
            os.remove(os.path.join(player_dir, player))
            print(f"Removed player {player}")
    players = top_players