

# set default values for the environment variables
export OMP_NUM_THREADS=8
if [ -z "$ADDR" ]
then
    export ADDR=127.0.0.1
fi

if [ -z "$WORLD_SIZE" ]
then
    export WORLD_SIZE=1
fi

if [ -z "$RANK" ]
then
    export RANK=0
fi

if [ -z "$MASTER_PORT" ]
then
    export MASTER_PORT=29501
fi

export WANDB_MODE=offline
accelerate_args="--config_file acce.yaml --num_machines $WORLD_SIZE \
                 --machine_rank $RANK --num_processes 1 \
                 --main_process_port $MASTER_PORT \
                 --main_process_ip $ADDR"
echo $accelerate_args

# train stage 1
accelerate launch $accelerate_args train.py --config configs/nf7_v3_SNR_rd_size_stroke_train.yaml \
    config.batch_size=1 \
    config.eval_interval=100


# train stage 2
# accelerate launch $accelerate_args train_stage2.py --config configs/stage2-v2-snr_train.yaml \
#     config.batch_size=1 \
#     config.eval_interval=100