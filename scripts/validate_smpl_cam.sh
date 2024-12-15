CONFIG=$1
CKPT=$2
PORT=${3:-65535}

HOST=$(hostname -i)

python ./scripts/validate_smpl_cam.py \
    --batch 32 \
    --gpus 1\
    --world-size 4 \
    --flip-test \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT}
