#!/bin/bash

# Root directory of the project
ROOT_DIR="~/LLaVA-OV-Manager"
MODEL_DIR="~/model"
DATA_DIR="~/data/LLaVA-OV-Manager"
CHECKPOINT_DIR=$MODEL_DIR/checkpoint/LLaVA-OV-Manager
mkdir $ROOT_DIR
mkdir $MODEL_DIR
mkdir $DATA_DIR
cd $ROOT_DIR

## Init Virtual Environment
pyenv virtualenv miniconda3-latest LLaVA-OV
conda activate LLaVA-OV
conda install python==3.10.12

## Install LLaVA
cd $ROOT_DIR
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
git reset --hard 79ef45a6d8b89b92d7a8525f077c3a3a9894a87d
cd LLaVA-NeXT
pip install --upgrade pip  # enable PEP 660 support
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
conda install nvidia/label/cuda-12.1.0::cuda
conda install nvidia/label/cuda-12.1.0::cuda-toolkit
pip install -e ".[train]"
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation

## Install lmms-eval
cd $ROOT_DIR
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
git reset --hard bb3fd824651336d6b001d86dd57a7042bf3bcf0b
cd lmms-eval
pip install -e .
pip install Levenshtein
pip install reka-api
pip install langdetect
pip install immutabledict
pip install fsspec==2023.10.0
# import nltk ; nltk.download('punkt_tab')
pip install datasets==2.20.0 # based on https://github.com/EvolvingLMMs-Lab/lmms-eval/issues/335, for offline evaluation


## [Manually] Cover the ROOT_DIR with our own code from https://github.com/LooperXX/LLaVA-OV-Manager


## Model Download
# export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download --resume-download Qwen/Qwen2-0.5B-Instruct --local-dir $MODEL_DIR/Qwen2-0.5B-Instruct --local-dir-use-symlinks False 

huggingface-cli download --resume-download google/siglip-so400m-patch14-384 --local-dir $MODEL_DIR/siglip-so400m-patch14-384 --local-dir-use-symlinks False 

huggingface-cli download --resume-download lmms-lab/llava-onevision-projectors --local-dir $MODEL_DIR/llava-onevision-projectors --local-dir-use-symlinks False

huggingface-cli download --resume-download lmms-lab/llavanext-qwen-siglip-tokenizer --local-dir $MODEL_DIR/llavanext-qwen-siglip-tokenizer --local-dir-use-symlinks False

huggingface-cli download --resume-download lmms-lab/llava-onevision-qwen2-0.5b-si --local-dir $MODEL_DIR/llava-onevision-qwen2-0.5b-si --local-dir-use-symlinks False

## Checkpoint Download
# export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download --resume-download LooperXX/LLaVA-OV-Manager --local-dir $CHECKPOINT_DIR --local-dir-use-symlinks False

## Data Download
# export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download --resume-download --repo-type dataset liuhaotian/LLaVA-Pretrain --local-dir $DATA_DIR/stage1/LLaVA-Pretrain --local-dir-use-symlinks False 

huggingface-cli download --resume-download --repo-type dataset lmms-lab/LLaVA-ReCap-558K --local-dir $DATA_DIR/stage1.5/LLaVA-ReCap-558K --local-dir-use-symlinks False 

huggingface-cli download --resume-download --repo-type dataset lmms-lab/LLaVA-ReCap-118K --local-dir $DATA_DIR/stage1.5/LLaVA-ReCap-118K --local-dir-use-symlinks False 

huggingface-cli download --resume-download --repo-type dataset lmms-lab/LLaVA-ReCap-CC3M --local-dir $DATA_DIR/stage1.5/LLaVA-ReCap-CC3M --local-dir-use-symlinks False 

huggingface-cli download --resume-download --repo-type dataset lmms-lab/LLaVA-OneVision-Mid-Data --local-dir $DATA_DIR/stage1.5/LLaVA-OneVision-Mid-Data --local-dir-use-symlinks False 

huggingface-cli download --resume-download --repo-type dataset lmms-lab/LLaVA-NeXT-Data --local-dir $DATA_DIR/stage2/LLaVA-NeXT-Data --local-dir-use-symlinks False 

huggingface-cli download --resume-download --repo-type dataset lmms-lab/LLaVA-OneVision-Data --local-dir $DATA_DIR/stage2/LLaVA-OneVision-Data --local-dir-use-symlinks False 

huggingface-cli download --resume-download --repo-type dataset Zhiqiang007/MathV360K --local-dir $DATA_DIR/stage2/MathV360K --local-dir-use-symlinks False 


## Data Processing
# Note: The main data processing steps are following the official LLaVA-NeXT repository. https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main/scripts/train.
# Note: We fix some bugs in the original training data of LLaVA-OneVision based on the open-source community's feedback and our own observations.

### Stage1
cd $DATA_DIR/stage1/LLaVA-Pretrain
unzip images.zip -d images

### Stage1.5
cd $DATA_DIR/stage1.5

#### Recap-* Data
python $ROOT_DIR/LLaVA-NeXT/scripts/manager/data/data_convert_stage1.5.py

#### LLaVA-OneVision-Mid-Data
cd LLaVA-OneVision-Mid-Data

cd synthdog_en
for file in *.tar.gz; do tar -xzf "$file"; done
mv synthdog-en ../../images
mkdir ../../json/synthdog_en
cp synthdog_en_processed.json ../../json/synthdog_en/synthdog_en_100k.json

cd ../synthdog_zh
for file in *.tar.gz; do tar -xzf "$file"; done
mv synthdog-zh ../../images
mkdir ../../json/synthdog_zh
cp synthdog_zh_processed.json ../../json/synthdog_zh/synthdog_zh_100k.json

cd ../ureader_tr
for file in *.tar.gz; do tar -xzf "$file"; done
mv docvqa ../../images
mv ureader-instruction-1.0 ../../images
cp ureader_tr_processed.json ../../json/ureader_tr_sft.json

cd ../evol_instruct 
cp evol_instruct_processed.json ../../json/Evol-Instruct-GPT4-Turbo-143K.json


### Stage2
cd $DATA_DIR/stage2

#### LLaVA-OneVision-Data
cd LLaVA-OneVision-Data
python $ROOT_DIR/LLaVA-NeXT/scripts/manager/data/data_convert_stage2.py

##### MathV360K (there are four parts that are not provided in LLaVA-OneVision-Data, then we extract from the original MathV360K dataset
cd ../MathV360K
unzip data_images.zip
python $ROOT_DIR/LLaVA-NeXT/scripts/manager/data/data_convert_stage2_MathV360K.py 

##### raven(cauldron) following https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data/discussions/11 to filter bad samples
python $ROOT_DIR/LLaVA-NeXT/scripts/manager/data/data_filter_stage2_raven.py

##### many datasets don't insert <image> into the text, fix them (including cauldron subsets (all) and lrv_normal (part of it) )
## https://huggingface.co/datasets/HuggingFaceM4/the_cauldron
python $ROOT_DIR/LLaVA-NeXT/scripts/manager/data/data_fix_stage2.py

##### ureader_kg and ureader_qa
cd ../ureader_kg
for file in *.tar.gz; do tar -xzf "$file"; done
mv docvqa ../../images
mv ureader-instruction-1.0 ../../images
cp ureader_kg_processed.json ../../json/ureader_kg_sft.json

cd ../ureader_qa
for file in *.tar.gz; do tar -xzf "$file"; done
cp -r  docvqa/* ../../images/docvqa/
rm -rf docvqa
cp -r  ureader-instruction-1.0/* ../../images/ureader-instruction-1.0/
rm -rf ureader-instruction-1.0
cp ureader_qa_processed.json ../../json/ureader_qa_sft.json

##### cambrian(filtered)
cd ../cambrian\(filtered\)
for file in *.tar.gz; do tar -xzf "$file"; done
mv cambrian_selection ../../images
cp cambrian\(filtered\)_processed.json ../../json/cambrian_filtered_gpt4vo_sp_token_fltd_max10k.json

##### Evol-Instruct
cp $DATA_DIR/stage1.5/json/Evol-Instruct-GPT4-Turbo-143K.json $DATA_DIR/stage2/json/Evol-Instruct-GPT4-Turbo-143000.json

#### LLaVA-NeXT-Data
cd ../../LLaVA-NeXT-Data/llava_next_raw_format
for file in *.tar.gz; do tar -xzf "$file"; done
mv ai2d ../../images
mv chartqa ../../images
mv coco ../../images
mv dvqa ../../images
mv gqa ../../images
mv laion-gpt4v-20231128 ../../images
mv llava_pretrain_lcs558k ../../images
mv ocr_vqa ../../images
mv sharegpt4v ../../images
mv synthdog-en ../../images
mv vg ../../images
cp -r docvqa/* ../../images/docvqa
rm -rf docvqa
cp llava_next_raw_format_processed.json ../../json/llava_next_fit_mix_filtered_text_wild_738590.json
