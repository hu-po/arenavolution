# directory containing directories with one player, directory name is player name
# player directory contains dict.yaml, and logs/, model/
# run to create player

import os
import yaml
from dataclasses import dataclass


@dataclass
class Organism:
    name: str
    elo: int
    wins: int
    losses: int
    dna: str # block of code
    epi: dict # epigenetics is hyperparameters or something?
    

def load_organism(name: str, base_org_dir: str = "/home/oop/dev/data/"):
    org_dir = os.path.join(base_org_dir, name)
    hparams = yaml.load(os.path.join(org_dir, "dict.yaml"))

     
def make_organism(name: str, base_org_dir: str = "/home/oop/dev/data/"):
    org_dir = os.path.join(base_org_dir, name)
    os.makedirs(org_dir, exist_ok=True)
    os.makedirs(os.path.join(org_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(org_dir, "model"), exist_ok=True)
    with open(os.path.join(org_dir, "dict.yaml"), "w") as f:
        yaml.dump({}, f)
    return Organism(name=name, elo=0, wins=0, losses=0, dna="", epi={})