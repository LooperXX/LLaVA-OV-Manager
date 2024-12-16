import yaml
import json
import math
import transformers
from tqdm import tqdm
from multiprocessing import Pool
import os

base_json_path = "~/data/LLaVA-OV-Manager/stage2/json"

for data_path in ["~/LLaVA-OV-Manager/LLaVA-NeXT/scripts/manager/main/stage2.yaml"]:
    # print("###########################")
    # print(f"Loading {data_path}")
    with open(data_path, "r") as file:
        yaml_data = yaml.safe_load(file)
        datasets = yaml_data.get("datasets")
        for dataset in datasets:
            json_path = dataset.get("json_path")
            sampling_strategy = dataset.get("sampling_strategy", "all")
            sampling_number = None

            # print(
                # f"Loading {json_path} with {sampling_strategy} sampling strategy")

            if json_path.endswith(".jsonl"):
                cur_data_dict = []
                with open(json_path, "r") as json_file:
                    for line in json_file:
                        cur_data_dict.append(json.loads(line.strip()))
            elif json_path.endswith(".json"):
                with open(json_path, "r") as json_file:
                    cur_data_dict = json.load(json_file)
            else:
                raise ValueError(f"Unsupported file type: {json_path}")

            # print(f"Loaded {len(cur_data_dict)} samples from {json_path}")

            def check_image(data):
                conversations = data["conversations"]
                if data.get("image") is None:
                    # return 0
                    return 1
                image_count = 0
                for conversation in conversations:
                    if "<image>" in conversation["value"]:
                        image_count += 1
                if image_count == 0:
                    return 0
                return 1

            with Pool(8) as p:
                counts = p.map(check_image, cur_data_dict)

            # if there is at least one sample without <image> token in the conversation but with image in the data
            if min(counts) == 0:
                # print(f"{json_path}, counts: {len(counts)}, {sum(counts)}")
                new_data_dict = []
                # add <image> to the first turn if all turns do not have <image>
                for data in cur_data_dict:
                    image_flag = False
                    for conversation in data["conversations"]:
                        if "<image>" in conversation["value"]:
                            new_data_dict.append(data)
                            image_flag = True
                            break
                    if not image_flag:
                        data["conversations"][0]["value"] = "<image>\n" + data["conversations"][0]["value"]
                        new_data_dict.append(data)
                with open(os.path.join(base_json_path, json_path[:-5] + "_fixed.json"), "w") as json_file:
                    json.dump(new_data_dict, json_file, indent=4, ensure_ascii=False)
                print(f"Fixed {json_path} to {json_path[:-5] + '_fixed.json'}")

# Note: we don't count the template tokens and image tokens

'''

'''
