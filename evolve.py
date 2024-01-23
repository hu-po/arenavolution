# seed from args
# num_players from args
# for num_players
    # load a random player from directory
    # train eval player
# remove losing player
# duplicate winning player with variation (use LLM to generate variations)
import os
import uuid
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_players", type=int, default=10)
args = parser.parse_args()

base_output_dir: str = "/home/oop/dev/data/"
session_id = str(uuid.uuid4())[:6]
output_dir = os.path.join(base_output_dir, f"evo_{session_id}")
os.makedirs(output_dir, exist_ok=True)

