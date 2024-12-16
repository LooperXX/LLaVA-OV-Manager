# For non-API tasks
bash lmms-eval/scripts/llava_ov_non_api_group.sh 8 # the number of GPUs to use
bash lmms-eval/scripts/llava_ov_reproduce_non_api_group.sh 8 "Baseline" # the number of GPUs to use, CHECKPOINT_NAME
bash lmms-eval/scripts/llava_ov_reproduce_non_api_group.sh 8 "Baseline + Grid"
bash lmms-eval/scripts/llava_ov_manager_non_api_group.sh  8 "Baseline + Manager"
bash lmms-eval/scripts/llava_ov_manager_non_api_group.sh  8 "Baseline + Grid + Manager"

# For API tasks

## evaluate one by one, and predict first (in a server maybe without network)), then evaluate on a server with network (do not need GPU)

### take the first task as an example, index \in [0, 6]
bash lmms-eval/scripts/llava_ov_api_predict_eval_from_log.sh 8 0 0 # the number of GPUs to use, the index of the task, 0 for predict only, 1 for evaluate only
bash lmms-eval/scripts/llava_ov_api_predict_eval_from_log.sh 8 0 1


bash lmms-eval/scripts/llava_ov_reproduce_api_predict_eval_from_log.sh 8 "Baseline" 0 0 # the number of GPUs to use, CHECKPOINT_NAME, the index of the task, 0 for predict only, 1 for evaluate only
bash lmms-eval/scripts/llava_ov_reproduce_api_predict_eval_from_log.sh 8 "Baseline" 0 1

bash lmms-eval/scripts/llava_ov_reproduce_api_predict_eval_from_log.sh 8 "Baseline + Grid" 0 0
bash lmms-eval/scripts/llava_ov_reproduce_api_predict_eval_from_log.sh 8 "Baseline + Grid" 0 1

bash lmms-eval/scripts/llava_ov_manager_api_predict_eval_from_log.sh 8 "Baseline + Manager" 0 0
bash lmms-eval/scripts/llava_ov_manager_api_predict_eval_from_log.sh 8 "Baseline + Manager" 0 1

bash lmms-eval/scripts/llava_ov_manager_api_predict_eval_from_log.sh 8 "Baseline + Grid + Manager" 0 0
bash lmms-eval/scripts/llava_ov_manager_api_predict_eval_from_log.sh 8 "Baseline + Grid + Manager" 0 1

## direct evaluate all tasks
### take the evaluation of the original model as an example
bash lmms-eval/scripts/llava_ov_api_direct.sh 8