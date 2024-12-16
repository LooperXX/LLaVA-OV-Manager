# Path
ROOT_DIR="~/LLaVA-OV-Manager"
TRAIN_CODE_DIR=$MODEL_DIR/LLaVA-NeXT
MODEL_DIR="~/model"
DATA_DIR="~/data/LLaVA-OV-Manager"
CHECKPOINT_DIR=$MODEL_DIR/checkpoint/LLaVA-OV-Manager

date
cd $TRAIN_CODE_DIR

# Wandb Environment Variables
export WANDB_ENTITY="YOUR_NAME"
export WANDB_PROJECT="LLaVA-OV-Manager"
export WANDB_MODE="online"
export WANDB_RUN_GROUP="stage1.5"
export WANDB_JOB_TYPE="train"
# export WANDB_API_KEY=YOUR_API_KEY
# export WANDB_NAME="My first run"
# export WANDB_NOTES="Smaller learning rate, more regularization."


# Hyper Parameters for Multi-GPU Training
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=0
export NCCL_SOCKET_IFNAME=ens81f0
export NCCL_DEBUG=INFO

NUM_GPUS=8
NNODES=1 # will affect per_device_train_batch_size and gradient_accumulation_steps for bsz 512
RANK=0
ADDR=$HOSTNAME
PORT=30098
MODEL_SCALE=0.5B #$1


# Model Scale and Path
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
if [ $MODEL_SCALE = '0.5B' ]
then
    LLM_VERSION="Qwen/Qwen2-0.5B-Instruct" # bsz 512
elif [ $MODEL_SCALE = '7B' ]
then
    LLM_VERSION="Qwen/Qwen2-7B-Instruct" # bsz 256
elif [ $MODEL_SCALE = '72B' ]
then
    LLM_VERSION="Qwen/Qwen2-72B-Instruct" # bsz 256
else
    echo "MODEL_SCALE not supported"
    exit 1
fi

## for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
## for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03

## Replace / with _ in model version for better naming
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
## To Load models from local directory
LLM_VERSION=${LLM_VERSION:5}
VISION_MODEL_VERSION=${VISION_MODEL_VERSION:7}

# Manager Part
STAGE1_MANAGER_LR=1e-3
MANAGER_LR=5e-5
MANAGER_TYPE=("static_zerouni" "adaptive_zerouni" "adaptive_last_zerouni" "adaptive_average_zerouni" "adaptive_cross_zerouni")
MANAGER_TYPE=${MANAGER_TYPE[0]}
MANAGER_GRID_TYPE="all"
MANAGER_VISION_SELECT_LAYERS_START=13
MANAGER_VISION_SELECT_LAYERS_INTERVAL=1
MANAGER_VISION_SELECT_LAYERS_END=26
MANAGER_INJECTION_START=0
MANAGER_INJECTION_INTERVAL=4
MANAGER_INJECTION_END=24
MANAGER_RESIDUAL=True

# Log Path and Rank
RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-stage1.5-manager-nogrid"
HYPER_NAME="${MANAGER_TYPE}-${MANAGER_GRID_TYPE}-${MANAGER_VISION_SELECT_LAYERS_START}-${MANAGER_VISION_SELECT_LAYERS_INTERVAL}-${MANAGER_VISION_SELECT_LAYERS_END}-${MANAGER_INJECTION_START}-${MANAGER_INJECTION_INTERVAL}-${MANAGER_INJECTION_END}-${MANAGER_RESIDUAL}"
RUN_NAME="${RUN_NAME}-${STAGE1_MANAGER_LR}-${MANAGER_LR}-${HYPER_NAME}"
LOG_PATH="$TRAIN_CODE_DIR/logs/main/$RUN_NAME.log"

if [ $NNODES = 1 ]
then
    RANK=0
    ADDR=$HOSTNAME
else
    ADDR=gpu02
    if [ $HOSTNAME = 'gpu02' ]
    then
        RANK=0
    elif [ $HOSTNAME = 'gpu03' ]
    then
        RANK=1
        LOG_PATH="$TRAIN_CODE_DIR/logs/main/gpu03.log"
    fi
fi


# Fine-tune the model

PROMPT_VERSION="qwen_1_5"
PREV_STAGE_CHECKPOINT="${CHECKPOINT_DIR}/llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-stage1-manager" # replace it with your last checkpoint training from mid stage
PREV_STAGE_CHECKPOINT="${PREV_STAGE_CHECKPOINT}-${STAGE1_MANAGER_LR}-${HYPER_NAME}"

echo "LOG_PATH: ${LOG_PATH}, RANK: ${RANK}, MODEL_SCALE: ${MODEL_SCALE}, PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}, RUN_NAME: ${RUN_NAME}, PROMPT_VERSION: ${PROMPT_VERSION}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2_fused_adamw.json \
    --model_name_or_path ${PREV_STAGE_CHECKPOINT} \
    --version ${PROMPT_VERSION} \
    --DATA_DIR ${TRAIN_CODE_DIR}/scripts/manager/main/stage1.5.yaml \
    --image_folder ${DATA_DIR}/stage1.5/images \
    --vision_tower ${MODEL_DIR}/${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_manager,mm_language_model" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_vision_tower_lr 2e-6 \
    --mm_manager_lr $MANAGER_LR \
    --group_by_modality_length True \
    --image_aspect_ratio square \
    --mm_patch_merge_type flat \
    --bf16 True \
    --output_dir ${CHECKPOINT_DIR}/${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --report_to wandb \
    --run_name $RUN_NAME \
    --mm_manager_type $MANAGER_TYPE \
    --mm_manager_grid_type $MANAGER_GRID_TYPE \
    --mm_manager_vision_select_layers_start $MANAGER_VISION_SELECT_LAYERS_START \
    --mm_manager_vision_select_layers_interval $MANAGER_VISION_SELECT_LAYERS_INTERVAL \
    --mm_manager_vision_select_layers_end $MANAGER_VISION_SELECT_LAYERS_END \
    --mm_manager_injection_start $MANAGER_INJECTION_START \
    --mm_manager_injection_interval $MANAGER_INJECTION_INTERVAL \
    --mm_manager_injection_end $MANAGER_INJECTION_END \
    --mm_manager_residual $MANAGER_RESIDUAL \
    > $LOG_PATH 2>&1
    # --attn_implementation sdpa \

# You can delete the sdpa attn_implementation if you want to use flash attn

date