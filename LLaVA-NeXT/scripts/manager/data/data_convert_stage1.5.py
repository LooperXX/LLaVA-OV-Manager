import os
from datasets import load_dataset
from tqdm import tqdm
import json

data_path = "~/data/LLaVA-OV-Manager/stage1.5"
for index in range(3):
    dataset_name = ["LLaVA-ReCap-558K", "LLaVA-ReCap-118K", "LLaVA-ReCap-CC3M"][index]
    json_file_name = ["blip558k_stage1.5_finetune.json", "coco118k_stage1.5_finetune.json", "cc3m_recap_data.json"][index]
    new_json_file_name = ["blip558k_stage1.5_finetune_w_prompt.json", "coco118k_stage1.5_finetune_w_prompt.json", "cc3m_recap_data_prompt_v2.json"][index]

    json_path = os.path.join(data_path, "json")
    image_folder = os.path.join(data_path, "images")
    os.makedirs(json_path, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(os.path.join(image_folder, dataset_name), exist_ok=True)

    # data = load_dataset(f'lmms-lab/{dataset_name}', cache_dir=os.path.join(data_path, dataset_name))
    print(dataset_name)
    data = load_dataset("parquet", data_dir=os.path.join(data_path, dataset_name))
    print(data)
    data = data["train"]

    converted_data = []

    for da in tqdm(data):
        json_data = {}
        json_data["id"] = da["id"]
        if da["image"] is not None:
            json_data["image"] = f"{dataset_name}/{da['id']}.jpg"
            da["image"].save(os.path.join(image_folder, json_data["image"]))
        json_data["conversations"] = da["conversations"]
        converted_data.append(json_data)


    with open(os.path.join(json_path, json_file_name), "w") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)


    # modify the json file to add an prompt behind <image> at each question
    # with open(os.path.join(json_path, json_file_name), "r") as f:
    #     converted_data = json.load(f)

    for da in tqdm(converted_data):
        da['conversations'][0]['value'] = f"{da['conversations'][0]['value']}\nPlease generate detailed descriptions of the given image."

    with open(os.path.join(json_path, new_json_file_name), "w") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)
