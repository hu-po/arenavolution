import os

def clean_docker():
    containers = os.popen("docker ps -aq").read().strip()
    if containers:
        os.system(f"docker kill {containers}")
        os.system(f"docker stop {containers}")
        os.system(f"docker rm {containers}")
    os.system("docker container prune -f")
