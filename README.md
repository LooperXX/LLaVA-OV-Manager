# LLaVA-OV-Manager

This repo is the official `Pytorch` implementation of the paper:

**Manager: Aggregating Insights from Unimodal Experts in Two-Tower VLMs and MLLMs**

[Xiao Xu](https://looperxx.github.io/), [Libo Qin](https://faculty.csu.edu.cn/qinlibo/), [Wanxiang Che](http://ir.hit.edu.cn/~car/), [Min-Yen Kan](https://www.comp.nus.edu.sg/~kanmy).

Under Review.

[Paper](https://looperxx.github.io/files/Manager.pdf) | [Code](https://github.com/LooperXX/LLaVA-OV-Manager) | [Model](https://huggingface.co/LooperXX/ManagerTower)

## Abstract

Two-Tower Vision--Language Models (VLMs) have demonstrated strong performance across various downstream VL tasks.
While BridgeTower further enhances performance by building bridges between encoders, it (i) suffers from ineffective layer-by-layer utilization of unimodal representations, (ii) restricts the flexible exploitation of different levels of unimodal semantic knowledge, and (iii) is limited to the evaluation on traditional low-resolution datasets only with the Two-Tower VLM architecture.
In this work, we propose Manager, a lightweight, efficient and effective plugin that adaptively aggregates insights from different levels of pre-trained unimodal experts to facilitate more comprehensive VL alignment and fusion.
First, under the Two-Tower VLM architecture, we introduce ManagerTower, a novel VLM that introduces the manager in each cross-modal layer.
No matter with or without VL Pre-training, ManagerTower outperforms previous strong baselines and achieves superior performance on 4 downstream VL tasks.
Moreover, we extend our exploration to the latest Multimodal Large Language Model (MLLM) architecture.
We demonstrate that LLaVA-OV-Manager significantly boosts the zero-shot performance of LLaVA-OV across different categories of capabilities, images, and resolutions on 20 downstream datasets, whether the multi-grid algorithm is enabled or not.
In-depth analysis reveals that both our manager and the multi-grid algorithm can be viewed as a plugin that improves the visual representation by capturing more diverse visual details from two orthogonal perspectives (depth and width).
Their synergy can mitigate the semantic ambiguity caused by the multi-grid algorithm and further improve performance.

## Checkpoints

We provide four model checkpoints for reproducing our results, including:

- Two reproduced versions of [LLaVA-OneVision-Qwen2-0.5b-si](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-si), indicated as `LLaVA-OV` for short.
- Two versions of our LLaVA-OneVision-Manager, indicated as `LLaVA-OV-Manager` for short.

You can download them from [here](https://huggingface.co/LooperXX/LLaVA-OV-Manager).

- [Baseline](https://huggingface.co/LooperXX/LLaVA-OV-Manager/tree/main/Baseline): `LLaVA-OV` without the Multi-Grid algorithm during training.
- [Baseline + Manager](https://huggingface.co/LooperXX/LLaVA-OV-Manager/tree/main/Baseline%20%2B%20Manager): `LLaVA-OV-Manager` without the Multi-Grid algorithm during training.
- [Baseline + Grid](https://huggingface.co/LooperXX/LLaVA-OV-Manager/tree/main/Baseline%20%2B%20Grid): `LLaVA-OV` with the Multi-Grid algorithm during training, which can be seen as a reproduction of the original [LLaVA-OneVision-Qwen2-0.5b-si](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-si).
- [Baseline + Grid + Manager](https://huggingface.co/LooperXX/LLaVA-OV-Manager/tree/main/Baseline%20%2B%20Grid%20%2B%20Manager): `LLaVA-OV-Manager` with the Multi-Grid algorithm during training.

## Deployment

- Run [`setup.sh`](https://github.com/LooperXX/LLaVA-OV-Manager/setup.sh) to set up the environment, prepare the pre-trained models, data and our checkpoints.
- [Optional] We use [wandb](https://wandb.ai/) to track experiments! Please remember to run `wandb login` and paste your token before running the script.

## Training

```Bash
cd LLaVA-NeXT

# Stage 1 for Baseline and Baseline + Grid
bash scripts/manager/main/stage1.sh

# Stage 1 for Baseline + Manager and Baseline + Grid + Manager
bash scripts/manager/main/stage1_manager.sh

# Baseline
bash scripts/manager/main/stage1.5_nogrid.sh
bash scripts/manager/main/stage2_nogrid.sh

# Baseline + Manager
bash scripts/manager/main/stage1.5_manager_nogrid.sh
bash scripts/manager/main/stage2_manager_nogrid.sh

# Baseline + Grid
bash scripts/manager/main/stage1.5.sh
bash scripts/manager/main/stage2.sh

# Baseline + Grid + Manager
bash scripts/manager/main/stage1.5_manager.sh
bash scripts/manager/main/stage2_manager.sh
```

## Evaluation

```Bash
cd lmms-eval

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
```

## Acknowledgement

This code repository is highly based on [LLaVA-One-Vision](https://github.com/LLaVA-VL/LLaVA-NeXT), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and [ManagerTower](https://github.com/LooperXX/ManagerTower).

