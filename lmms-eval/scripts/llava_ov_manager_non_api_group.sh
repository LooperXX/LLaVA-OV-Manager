export HF_DATASETS_OFFLINE=1

ROOT_DIR="~/LLaVA-OV-Manager"
MODEL_DIR="~/model"
NUM_PROCESSES=$1
CHECKPOINT_DIR=$MODEL_DIR/checkpoint/LLaVA-OV-Manager
CHECKPOINT_NAME=$2


tasks="mmmu_val,scienceqa_img,realworldqa,ai2d,chartqa,ok_vqa,textvqa_val,gqa,docvqa_test,infovqa_test,ocrbench,seedbench,vqav2_val"

for task in ${tasks//,/ } ; do
    echo $task
    date
    
    accelerate launch --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision_ours_manager \
    --output_path $ROOT_DIR/lmms-eval/logs \
    --model llava_onevision \
    --model_args pretrained=$CHECKPOINT_DIR/$CHECKPOINT_NAME,conv_template=qwen_1_5,model_name=llava_qwen_manager \
    --tasks $task
    date
done
