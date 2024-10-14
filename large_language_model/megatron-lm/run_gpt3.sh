#!/bin/bash

#SBATCH -A mlperf -t 23:59:59 --nodelist=worker-[0-7] --ntasks-per-node=8 --gpus=64 --cpus-per-task=10 --mem-per-cpu=14GB --job-name=mlperf-megatron

# Vars without defaults
LOG_DIR=${1:?LOG_DIR not set}
BPE_DIR=${2:?BPE_DIR not set}
CONT="${3:?CONT not set}"

# Vars with defaults
: "${MEGATRON_DIR:=$PWD}"
: "${GBS:=1536}"
: "${LR:=2.0e-5}"
: "${MIN_LR:=2.0e-6}"
: "${EVAL_INTERVAL:=$(( (24576 + ${GBS} - 1) / ${GBS} ))}"
: "${USE_BF16:=false}"  # set to false for FP32
: "${EXTERNAL_MODEL_CHECKPOINT_DIR:=}"
: "${EXTERNAL_TRAINING_ITERATIONS:=4000}"
: "${EXTERNAL_GBS:=1536}"

# Setup directories
CHECKPOINT_DIR="${LOG_DIR}/GPT3-175B-checkpoints"
TENSORBOARD_DIR="${LOG_DIR}/GPT3-175B-tensorboard"

mkdir -p ${LOG_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${TENSORBOARD_DIR}

export NCCL_TOPO_FILE=/var/run/nvidia-topologyd/virtualTopology.xml
export COM_DIR=/gpt3/dataset/preprocessed_c4_spm
export USE_BF16=true
export WORLD_SIZE=64
export EXTERNAL_MODEL_CHECKPOINT_DIR=/gpt3/checkpoint/bf16/ckpt4000
export MASTER_ADDR="10.0.2.216"
export MASTER_PORT=7000
export TORCH_CUDA_ARCH_LIST="9.0"
export NCCL_SOCKET_IFNAME="eth0"

#export NCCL_TIMEOUT=540
#export NCCL_LAUNCH_MODE=PARALLEL
#export NCCL_ASYNC_ERROR_HANDLING=1

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO

#export NCCL_IBEXT_DISABLE=1
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=1

# Get the data blend
. $PWD/gpt3_blend.sh

################################################################################
### Set exit duration based on variable time allocated for this specific job ###
# Query Slurm for the remaining job time left in the format [days-]hh:mm:ss
# format and pass the time (in units of minutes) to Megatron using variable
# EXIT_DURATION. The actual value passed is actually 13 minutes less for time
# to save model and extra margin. For our purposes we assume the days field
# will never be present to make parsing in bash easier. Note that setting
# EXIT_DURATION to 0 will terminate the job after 1 iteration.
timeleft=`squeue -j ${SLURM_JOBID} --noheader --format=%L`
timeleft=(`echo $timeleft | tr ':' ' '`)
EXIT_DURATION=$((timeleft[0]*60 + timeleft[1] - 15))
echo "setting exit duration to $EXIT_DURATION minutes"
################################################################################

options=" \
--num-workers 16 \
--exit-duration-in-mins ${EXIT_DURATION} \
--tensor-model-parallel-size 8 \
--pipeline-model-parallel-size 8 \
--sequence-parallel \
--recompute-activations \
--num-layers 96 \
--hidden-size 12288 \
--num-attention-heads 96 \
--seq-length 2048 \
--max-position-embeddings 2048 \
--micro-batch-size 1 \
--global-batch-size ${GBS} \
--train-samples 20000000 \
--lr-warmup-samples 407040 \
--lr-decay-samples 166809600 \
--lr ${LR} \
--min-lr ${MIN_LR} \
--lr-decay-style cosine \
--log-interval 1 \
--eval-iters -1 \
--eval-interval ${EVAL_INTERVAL} \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--train-data-path ${DATA_BLEND} \
--valid-data-path ${VALID_DATA_BLEND} \
--vocab-file ${BPE_DIR}/vocab.json \
--merge-file ${BPE_DIR}/merges.txt \
--save-interval 500 \
--save ${CHECKPOINT_DIR} \
--do-layernorm-bias-weight-decay \
--no-scaled-init \
--loss-scale 1.0 \
--split 100,0,0 \
--clip-grad 1.0 \
--weight-decay 0.1 \
--adam-beta1 0.9 \
--adam-beta2 0.95 \
--init-method-std 0.006 \
--log-params-norm \
--log-num-zeros-in-grad \
--log-validation-ppl-to-tensorboard \
--DDP-impl local \
--tensorboard-dir ${TENSORBOARD_DIR} \
--no-query-key-layer-scaling \
--no-seq-len-plus-one-tokens \
--seed ${RANDOM} "

[ ${USE_BF16} = true ] && options+=" --bf16"
if [ -n "${EXTERNAL_MODEL_CHECKPOINT_DIR}" ]; then
  options+=" \
		--no-load-rng \
		--no-load-optim \
		--use-ext-ckpt \
		--ext-iterations $(( $EXTERNAL_TRAINING_ITERATIONS * $EXTERNAL_GBS / $GBS)) \
		--ext-lr-steps $(( $EXTERNAL_TRAINING_ITERATIONS * $EXTERNAL_GBS)) \
		--load ${EXTERNAL_MODEL_CHECKPOINT_DIR}"
else
  options+=" --load ${CHECKPOINT_DIR}"
fi

echo "Job runs on the following nodes: ${SLURM_JOB_NODELIST}"

echo "PyTorch options: ${options}"

# Run
debug_cmd='echo "START training step NODE_ID:" $SLURM_NODEID "NODE_NAME:" $SLURMD_NODENAME "JOB_ID:" $SLURM_JOBID "RANK:" $SLURM_PROCID "TASK_ID:" $SLURM_LOCALID "PID:" $SLURM_TASK_PID "GPUs:" $CUDA_VISIBLE_DEVICES'
run_cmd="python -u ${MEGATRON_DIR}/pretrain_gpt.py ${options}"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

srun -l \
     --container-image $CONT \
     --container-mounts "$NCCL_TOPO_FILE:$NCCL_TOPO_FILE,$PWD:$PWD,${COM_DIR}:${COM_DIR},${LOG_DIR}:${LOG_DIR},${BPE_DIR}:${BPE_DIR},${EXTERNAL_MODEL_CHECKPOINT_DIR}:${EXTERNAL_MODEL_CHECKPOINT_DIR}" \
     --container-writable \
     --container-name="mlperf_megatron" \
     --export=ALL \
     --output=$LOG_DIR/megatron-%j.log sh -c "${debug_cmd} && ${run_cmd}"
#     --output=$LOG_DIR/megatron-%j-%N.log sh -c "${debug_cmd} && ${run_cmd}"

set +x

