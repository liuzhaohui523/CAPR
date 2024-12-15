EXPID=$1
CONFIG=$2
PORT=${3:-12345}

HOST=$(hostname -i)

export CUDA_VISIBLE_DEVICES=0,1,2,3

python ./scripts/train_smpl_cam.py \
    --nThreads 8 \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
    --exp-id ${EXPID} \
    --cfg ${CONFIG} --seed 88