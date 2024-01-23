# directory containing directories with one player, directory name is player name
# player directory contains dict.yaml, and logs/, model/
# run to create player

import os
import yaml
from dataclasses import dataclass

from gpt import make_variants

ORGANISM_DIR: str = os.environ.get("ORG_DIR", "/home/oop/dev/data/")
HPARAMS_FILENAME: str = "hparams.yaml"

@dataclass
class Organism:
    name: str
    code: str # Code block should define a model
    wins: int
    losses: int
    

def load_organism(name: str) -> Organism:
    org_dir = os.path.join(ORGANISM_DIR, name)
    assert os.path.isdir(org_dir), f"Organism {name} not found"
    hparams_filepath = os.path.join(org_dir, HPARAMS_FILENAME)
    assert os.path.isfile(hparams_filepath), f"Organism {name} missing hparams file"
    with open(hparams_filepath, "r") as file:
        hparams = yaml.load(file, Loader=yaml.FullLoader)
        organism = Organism(**hparams)
    return organism

     
def make_organism(name: str, hparams: dict) -> Organism:
    org_dir = os.path.join(ORGANISM_DIR, name)
    os.makedirs(org_dir, exist_ok=True)
    logs_dir = os.path.join(org_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    ckpt_dir = os.path.join(org_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    hparams_filepath = os.path.join(org_dir, HPARAMS_FILENAME)
    organism = Organism(name=name, **hparams)
    with open(hparams_filepath, "w") as f:
        yaml.dump(hparams, f)
    return organism

def reproduce(organisms: List[Organism], num_children: int = 5) -> List[Organism]:
    children = []
    for org in organisms:
        for i in range(num_children):
            child_name = f"{org.name}_{i}"
            child = make_organism(child_name, org.hparams)
            children.append(child)
    return children