export HF_DATASETS_OFFLINE=1
export API_TYPE=openai
export OPENAI_API_URL=https://api.openai.com/v1/chat/completions
export OPENAI_API_KEY="YOUR_API_KEY"

ROOT_DIR="~/LLaVA-OV-Manager"
MODEL_DIR="~/model"
NUM_PROCESSES=$1
CHECKPOINT_NAME=llava-onevision-qwen2-0.5b-si

tasks=("mathvista_testmini" "mmvet" "dc100_en" "llava_in_the_wild" "llava_wilder_small" "live_bench_2407" "live_bench_2409")

date

if [ $3 -eq 0 ]
then
  
    accelerate launch --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --batch_size 1 \
    --predict_only \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --output_path $ROOT_DIR/lmms-eval/logs \
    --model llava_onevision \
    --model_args pretrained=$MODEL_DIR/$CHECKPOINT_NAME,conv_template=qwen_1_5,model_name=llava_qwen \
    --tasks ${tasks[$2]}
else

    accelerate launch --num_processes=1 \
    -m lmms_eval \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --output_path $ROOT_DIR/lmms-eval/logs \
    --model from_log \
    --model_args pretrained=$MODEL_DIR/$CHECKPOINT_NAME,conv_template=qwen_1_5,model_name=llava_qwen,logs=$ROOT_DIR/lmms-eval/logs/LLaVA-OV-Manager__$CHECKPOINT_NAME,evaltask=${tasks[$2]} \
    --tasks ${tasks[$2]}

fi

date