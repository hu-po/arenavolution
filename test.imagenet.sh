export DATA_PATH="/home/oop/dev/data"
export CKPT_PATH="/home/oop/dev/test/ckpt"
export LOGS_PATH="/home/oop/dev/test/logs"
docker build \
     -t "imagenet_pytorch" \
     -f Dockerfile .
docker run \
    -it \
    --rm \
    -p 5555:5555 \
    --gpus 0 \
    -v ${DATA_PATH}:/data \
    -v ${CKPT_PATH}:/ckpt \
    -v ${LOGS_PATH}:/logs \
    imagenet_pytorch