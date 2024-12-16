import json
from PIL import Image
import os


json_path = "~/data/LLaVA-OV-Manager/stage2/json/raven_42000.json"
new_json_path = "~/data/LLaVA-OV-Manager/stage2/json/raven_42000_filtered.json"
image_path = "~/data/LLaVA-OV-Manager/stage2/images"

with open(json_path, "r") as f:
    data = json.load(f)

# filter the data with image that < 500px, following https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data/discussions/11

filtered_data = []
for item in data:
    item_image_path = os.path.join(image_path, item["image"])
    # get the image size
    im = Image.open(item_image_path)
    width, height = im.size
    if width < 500 or height < 500:
        continue
    filtered_data.append(item)

print(f"origin data: {len(data)}, filtered data: {len(filtered_data)}, filtered rate: {len(filtered_data)/len(data)}")

with open(new_json_path, "w") as f:
    json.dump(filtered_data, f, indent=4, ensure_ascii=False)
