import os
import uuid
import argparse
import yaml
from typing import List
import subprocess
import shutil

from gpt import make_variant, make_player_name
from utils import clean_docker

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_players", type=int, default=10)
parser.add_argument("--base_dir", type=str, default="/home/oop/dev/data/")
parser.add_argument("--data_dir", type=str, default="/home/oop/dev/data/centipede_chickadee")
args = parser.parse_args()


def load_player_from_dir(name: str, player_dir: str) -> dict:
    org_dir = os.path.join(player_dir, name)
    assert os.path.isdir(org_dir), f"Player {name} not found"
    hparams_filepath = os.path.join(org_dir, "hparams.yaml")
    assert os.path.isfile(hparams_filepath), f"Player {name} missing hparams file"
    with open(hparams_filepath, "r") as file:
        hparams = yaml.load(file, Loader=yaml.FullLoader)
    with open(os.path.join(org_dir, "model.py"), "r") as file:
        hparams["model_code"] = file.read()
    return hparams


def write_player_to_dir(hparams: dict, player_dir: str) -> dict:
    org_dir = os.path.join(player_dir, hparams["name"])
    os.makedirs(org_dir, exist_ok=True)
    logs_dir = os.path.join(org_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    ckpt_dir = os.path.join(org_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    hparams_filepath = os.path.join(org_dir, "hparams.yaml")
    with open(hparams_filepath, "w") as f:
        yaml.dump(hparams, f)
    model_filepath = os.path.join(org_dir, "model.py")
    with open(model_filepath, "w") as f:
        f.write(hparams["model_code"])


def reproduce(players: List[dict], player_dir:str, num_children: int = 1) -> List[dict]:
    code_blocks = [p['model_code'] for p in players]
    children = []
    for _ in range(num_children):
        child = write_player_to_dir(
            {
                "name": make_player_name(),
                "model_code": make_variant(code_blocks),
            },
            player_dir,
        )
        children.append(child)
    return children


def run_traineval(player: dict, player_dir: str):
    org_dir = os.path.join(player_dir, player["name"])
    model_filepath = os.path.join(org_dir, "model.py")
    ckpt_filepath = os.path.join(org_dir, "ckpt")
    logs_filepath = os.path.join(org_dir, "logs")
    clean_docker()
    docker_process = subprocess.Popen(
        [
            "docker",
            "run",
            "--rm",
            # "-p 5555:5555",
            "--gpus 0",
            f"-v {model_filepath}:/src/model.py",
            f"-v {ckpt_filepath}:/ckpt",
            f"-v {logs_filepath}:/logs",
            f"-v {args.data_dir}:/data",
            "evolver",
        ]
    )
    # Check to see if docker process dies due to model error
    # lots of error checking, return tuple with failure boolean
    return docker_process


print("Starting evolution")
session_id = str(uuid.uuid4())[:6]
output_dir = os.path.join(args.base_dir, f"evo_{session_id}")
os.makedirs(output_dir, exist_ok=True)
print(f"Saving output to {output_dir}")
player_dir = os.path.join(output_dir, "players")

# Copy over the starter players in local directory
starter_player_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "players")
shutil.copytree(starter_player_dir, player_dir)
players = [
    load_player_from_dir("mlpee", player_dir),
    load_player_from_dir("convy", player_dir),
]
players += reproduce(players, player_dir)
for player in players:
    print(f"Running traineval for {player['name']}")
    print("--------------------")
    print(player['model_code'])
    print("--------------------")
    run_traineval(player, player_dir)
