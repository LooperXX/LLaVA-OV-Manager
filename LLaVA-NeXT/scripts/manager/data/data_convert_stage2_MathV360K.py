import os
from datasets import load_dataset
from tqdm import tqdm
import json
import shutil

# since LLaVA-OneVision-Data do not provide MathV360K_TQA_10181.json, MathV360K_PlotQA_5485.json, MathV360K_VQA-AS_5907.json, MathV360K_VQA-RAD_2130.json
# we need to extract them from original MathV360K dataset https://huggingface.co/datasets/Zhiqiang007/MathV360K

data_path = "~/data/LLaVA-OV-Manager/stage2/MathV360K"
image_path = "~/data/LLaVA-OV-Manager/stage2/images"
old_image_path = "~/data/LLaVA-OV-Manager/stage2/MathV360K/data_images"
json_path = "~/data/LLaVA-OV-Manager/stage2/json"
json_file_name_dict = {
    "TQA": "MathV360K_TQA_10181.json",
    "PlotQA": "MathV360K_PlotQA_5485.json",
    "VQA-AS": "MathV360K_VQA-AS_5907.json",
    "VQA-RAD": "MathV360K_VQA-RAD_2130.json"
}

# read train_samples_all_tuning.json
with open(os.path.join(data_path, "train_samples_all_tuning.json"), "r") as f:
    all_data = json.load(f)

for dataset_name, json_file_name in json_file_name_dict.items():
    new_dataset_name = f"{dataset_name}(MathV360K)"
    os.makedirs(os.path.join(image_path, new_dataset_name), exist_ok=True)

    cur_data = []
    for data in all_data:
        if data["image"].startswith(dataset_name):
            cur_data.append(data)
            old_data_image_path = os.path.join(old_image_path, data["image"])
            cur_image_format = data["image"].split(".")[-1]
            new_data_image_path = os.path.join(image_path, new_dataset_name, f"{data['id']}.{cur_image_format}")
            shutil.copy(old_data_image_path, new_data_image_path)
            cur_data[-1]["image"] = f"{new_dataset_name}/{data['id']}.{cur_image_format}"

    print(f"{dataset_name} with {len(cur_data)} samples")

    with open(os.path.join(json_path, json_file_name), "w") as f:
        json.dump(cur_data, f, indent=4, ensure_ascii=False)
