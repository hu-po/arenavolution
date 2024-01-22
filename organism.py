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
    

def load_org(name: str, base_org_dir: str = "/home/oop/dev/data/"):
    org_dir = os.path.join(base_org_dir, name)
    hparams = yaml.load(os.path.join(org_dir, "dict.yaml"))

     