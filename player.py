# directory containing directories with one player, directory name is player name
# player directory contains dict.yaml, and logs/, model/
# run to create player

from dataclasses import dataclass


@dataclass
class Player:
    name: str
    elo: int
    wins: int
    losses: int
    hparams: dict
    

if __name__ == "__main__":
    pass