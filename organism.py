# directory containing directories with one player, directory name is player name
# player directory contains dict.yaml, and logs/, model/
# run to create player

import os
import yaml
from typing import List
import subprocess
from dataclasses import dataclass

from gpt import make_variant, make_organism_name
from utils import clean_docker

ORGANISM_DIR: str = os.environ.get("ORG_DIR", "/home/oop/dev/data/")
HPARAMS_FILENAME: str = "hparams.yaml"
MODEL_FILENAME: str = "model.py"
CKPT_DIR_NAME = "ckpt"
LOGS_DIR_NAME = "logs"


@dataclass
class Organism:
    name: str
    code: str  # Code block should define a model
    wins: int
    losses: int


def load_organism_from_dir(name: str) -> Organism:
    org_dir = os.path.join(ORGANISM_DIR, name)
    assert os.path.isdir(org_dir), f"Organism {name} not found"
    hparams_filepath = os.path.join(org_dir, HPARAMS_FILENAME)
    assert os.path.isfile(hparams_filepath), f"Organism {name} missing hparams file"
    with open(hparams_filepath, "r") as file:
        hparams = yaml.load(file, Loader=yaml.FullLoader)
        organism = Organism(**hparams)
    return organism


def make_organism(hparams: dict) -> Organism:
    org_dir = os.path.join(ORGANISM_DIR, hparams["name"])
    os.makedirs(org_dir, exist_ok=True)
    logs_dir = os.path.join(org_dir, LOGS_DIR_NAME)
    os.makedirs(logs_dir, exist_ok=True)
    ckpt_dir = os.path.join(org_dir, CKPT_DIR_NAME)
    os.makedirs(ckpt_dir, exist_ok=True)
    hparams_filepath = os.path.join(org_dir, HPARAMS_FILENAME)
    organism = Organism(**hparams)
    with open(hparams_filepath, "w") as f:
        yaml.dump(hparams, f)
    model_filepath = os.path.join(org_dir, MODEL_FILENAME)
    with open(model_filepath, "w") as f:
        f.write(hparams["code"])
    return organism


def reproduce(organisms: List[Organism], num_children: int = 5) -> List[Organism]:
    code_blocks = [org.code for org in organisms]
    children = []
    for _ in range(num_children):
        child = make_organism(
            {
                "name": make_organism_name(),
                "code": make_variant(code_blocks),
                "wins": 0,
                "losses": 0,
            }
        )
        children.append(child)
    return children


def spawn(organism: Organism):
    org_dir = os.path.join(ORGANISM_DIR, organism.name)
    model_filepath = os.path.join(org_dir, MODEL_FILENAME)
    ckpt_filepath = os.path.join(org_dir, CKPT_DIR_NAME)
    logs_filepath = os.path.join(org_dir, LOGS_DIR_NAME)
    clean_docker()
    docker_process = subprocess.Popen(
        [
            "docker",
            "run",
            "--rm",
            "-p 5555:5555",
            "--gpus 0",
            f"-v {model_filepath}:/src/model.py"
            f"-v {ckpt_filepath}:/ckpt",
            f"-v ${logs_filepath}:/logs",
            "imagenet_pytorch",
        ]
    )
    # Check to see if docker process dies due to model error
    # lots of error checking, return tuple with failure boolean
    return docker_process
