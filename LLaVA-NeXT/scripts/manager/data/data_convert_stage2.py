import os
from datasets import load_dataset
from tqdm import tqdm
import json

data_path = "~/data/LLaVA-OV-Manager/stage2"
json_path = os.path.join(data_path, "json")
image_folder = os.path.join(data_path, "images")
os.makedirs(json_path, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)

# get all directory names in the data_path
data_path = f"{data_path}/LLaVA-OneVision-Data"
dataset_name_list = os.listdir(data_path)
# remove files and '.cache', '.gitattributes'
dataset_name_list = [name for name in dataset_name_list if os.path.isdir(os.path.join(data_path, name)) and not name.startswith('.')]
# remove ureader_kg and ureader_qa
dataset_name_list.remove("ureader_kg")
dataset_name_list.remove("ureader_qa")
dataset_name_list.remove("cambrian(filtered)")
# sort the dataset_name_list
dataset_name_list.sort()
json_file_name_dict = dict.fromkeys(dataset_name_list)
print(json_file_name_dict)

# map the dataset_name to the json_file_name from official LLaVA-OneVision training data
json_file_name_dict["CLEVR-Math(MathV360K)"] = "MathV360K_CLEVR-Math_5290.json"
json_file_name_dict["FigureQA(MathV360K)"] = "MathV360K_FigureQA_17597.json"
json_file_name_dict["GEOS(MathV360K)"] = "MathV360K_GEOS_508.json"
json_file_name_dict["GeoQA+(MathV360K)"] = "MathV360K_GeoQA+_17172.json"
json_file_name_dict["Geometry3K(MathV360K)"] = "MathV360K_Geometry3K_9734.json"
json_file_name_dict["IconQA(MathV360K)"] = "MathV360K_IconQA_22599.json"
json_file_name_dict["MapQA(MathV360K)"] = "MathV360K_MapQA_5235.json"
json_file_name_dict["PMC-VQA(MathV360K)"] = "MathV360K_PMC-VQA_35958.json"
json_file_name_dict["Super-CLEVR(MathV360K)"] = "MathV360K_Super-CLEVR_8652.json"
json_file_name_dict["TabMWP(MathV360K)"] = "MathV360K_TabMWP_22462.json"
json_file_name_dict["UniGeo(MathV360K)"] = "MathV360K_UniGeo_11959.json"
json_file_name_dict["VisualWebInstruct(filtered)"] = "VisualWebInstruct_filtered_263589.json"
json_file_name_dict["VizWiz(MathV360K)"] = "MathV360K_VizWiz_6614.json"
json_file_name_dict["ai2d(cauldron,llava_format)"] = "ai2d_llava_format_2434.json"
json_file_name_dict["ai2d(gpt4v)"] = "ai2d_azuregpt_detailed_understanding_4874.json"
json_file_name_dict["ai2d(internvl)"] = "ai2d_train_internvl_single_12413.json"
json_file_name_dict["allava_instruct_laion4v"] = "allava_instruct_laion4v_50000.json"
json_file_name_dict["allava_instruct_vflan4v"] = "allava_instruct_vflan4v_20000.json"
json_file_name_dict["aokvqa(cauldron,llava_format)"] = "aokvqa_16539_llava_format.json"
json_file_name_dict["chart2text(cauldron)"] = "chart2text_26961.json"
json_file_name_dict["chartqa(cauldron,llava_format)"] = "chartqa_18265_llava_format.json"
json_file_name_dict["chrome_writting"] = "chrome_writting_train_8835.json"
json_file_name_dict["clevr(cauldron,llava_format)"] = "clevr_70000_llava_format.json"
json_file_name_dict["diagram_image_to_text(cauldron)"] = "diagram_image_to_text_300.json"
json_file_name_dict["dvqa(cauldron,llava_format)"] = "dvqa_200000_llava_format.json"
json_file_name_dict["figureqa(cauldron,llava_format)"] = "figureqa_100000_llava_format.json"
json_file_name_dict["geo170k(align)"] = "geo170k_align_converted_60252.json"
json_file_name_dict["geo170k(qa)"] = "geo170k_qa_converted_67833.json"
json_file_name_dict["geo3k"] = "geo3k_2101.json"
json_file_name_dict["geomverse(cauldron)"] = "geomverse_9303.json"
json_file_name_dict["hateful_memes(cauldron,llava_format)"] = "hateful_memes_8500_llava_format.json"
json_file_name_dict["hitab(cauldron,llava_format)"] = "hitab_2500_llava_format.json"
json_file_name_dict["hme100k"] = "hme100k_train_clean_74502.json"
json_file_name_dict["iam(cauldron)"] = "iam_5663.json"
json_file_name_dict["iconqa(cauldron,llava_format)"] = "iconqa_llava_format_27307.json"
json_file_name_dict["iiit5k"] = "iiit5k_annotations_2000.json"
json_file_name_dict["image_textualization(filtered)"] = "image_textualization_dataset_filtered.json"
json_file_name_dict["infographic(gpt4v)"] = "infographic_azuregpt4v_1992.json"
json_file_name_dict["infographic_vqa"] = "infographic_vqa_2118_llava_format.json"
json_file_name_dict["infographic_vqa_llava_format"] = "infographic_vqa_4404.json"
json_file_name_dict["intergps(cauldron,llava_format)"] = "intergps_1280_llava_format.json"
json_file_name_dict["k12_printing"] = "k12_printing_train_256646.json"
json_file_name_dict["llavar_gpt4_20k"] = "llavar_gpt4_20k.json"
json_file_name_dict["lrv_chart"] = "lrv_chart_1787.json"
json_file_name_dict["lrv_normal(filtered)"] = "lrv_normal_gpt4v_filtered_10500.json"
json_file_name_dict["magpie_pro(l3_80b_mt)"] = "magpie_pro_l3_80b_mt_300000_sp_token_fltd_299998.json"
json_file_name_dict["magpie_pro(l3_80b_st)"] = "magpie_pro_l3_80b_st_300000.json"
json_file_name_dict["magpie_pro(qwen2_72b_st)"] = "magpie_pro_qwen2_72b_st_300000_sp_token_fltd_299992.json"
json_file_name_dict["mapqa(cauldron,llava_format)"] = "mapqa_37417_llava_format.json"
json_file_name_dict["mathqa"] = "mathqa_29837.json"
json_file_name_dict["mavis_math_metagen"] = "mavis_math_metagen_87358.json"
json_file_name_dict["mavis_math_rule_geo"] = "mavis_math_rule_geo_100000.json"
json_file_name_dict["multihiertt(cauldron)"] = "multihiertt_7619.json"
json_file_name_dict["orand_car_a"] = "orand_car_a_train_2009.json"
json_file_name_dict["raven(cauldron)"] = "raven_42000.json"
json_file_name_dict["rendered_text(cauldron)"] = "rendered_text_10000.json"
json_file_name_dict["robut_sqa(cauldron)"] = "robut_sqa_8514.json"
json_file_name_dict["robut_wikisql(cauldron)"] = "robut_wikisql_74989.json"
json_file_name_dict["robut_wtq(cauldron,llava_format)"] = "robut_wtq_38246_llava_format.json"
json_file_name_dict["scienceqa(cauldron,llava_format)"] = "scienceqa_llava_format_4976.json"
json_file_name_dict["scienceqa(nona_context)"] = "scienceqa_nona_context_19218.json"
json_file_name_dict["screen2words(cauldron)"] = "screen2words_15730.json"
json_file_name_dict["sharegpt4o"] = "sharegpt4o_dataset.json"
json_file_name_dict["sharegpt4v(coco)"] = "sharegpt4v-coco-50k.json"
json_file_name_dict["sharegpt4v(knowledge)"] = "sharegpt4v-knowledge-2k.json"
json_file_name_dict["sharegpt4v(llava)"] = "sharegpt4v-llava-30k.json"
json_file_name_dict["sharegpt4v(sam)"] = "sharegpt4v-sam-20k.json"
json_file_name_dict["sroie"] = "sroie_data_33626.json"
json_file_name_dict["st_vqa(cauldron,llava_format)"] = "st_vqa_17247_llava_format.json"
json_file_name_dict["tabmwp(cauldron)"] = "tabmwp_22722.json"
json_file_name_dict["tallyqa(cauldron,llava_format)"] = "tallyqa_98680_llava_format.json"
json_file_name_dict["textcaps"] = "textcaps_train_21952.json"
json_file_name_dict["textocr(gpt4v)"] = "textocr_gpt4v_train_converted_25114.json"
json_file_name_dict["tqa(cauldron,llava_format)"] = "tqa_llava_format_27307.json"
json_file_name_dict["ureader_cap"] = "ureader_cap_sft.json"
json_file_name_dict["ureader_ie"] = "ureader_ie_sft.json"
json_file_name_dict["vision_flan(filtered)"] = "vision_flan_filtered_186070.json"
json_file_name_dict["vistext(cauldron)"] = "vistext_9969.json"
json_file_name_dict["visual7w(cauldron,llava_format)"] = "visual7w_llava_format_14366.json"
json_file_name_dict["visualmrc(cauldron)"] = "visualmrc_3027.json"
json_file_name_dict["vqarad(cauldron,llava_format)"] = "vqarad_313_llava_format.json"
json_file_name_dict["vsr(cauldron,llava_format)"] = "vsr_2157_llava_format.json"
json_file_name_dict["websight(cauldron)"] = "websight_10000.json"


# convert the dataset
# count = 0
for dataset_name, json_file_name in json_file_name_dict.items():
    cauldron_flag = False
    # if count < 76:
    #     count += 1
    #     continue
    print(dataset_name)
    if 'cauldron' in dataset_name or "infographic_vqa_llava_format" == dataset_name:
        cauldron_flag = True

    data = load_dataset("parquet", data_dir=os.path.join(data_path, dataset_name))
    os.makedirs(os.path.join(image_folder, dataset_name), exist_ok=True)
    print(data)
    data = data["train"]

    converted_data = []

    for da in tqdm(data):
        json_data = {}
        json_data["id"] = da["id"]
        if da["image"] is not None:
            if cauldron_flag:
                # extract "websight_00007569" from "cauldron/websight/images/websight_00007569.png"
                json_data["image"] = f"{dataset_name}/{da['id'].rsplit('/')[-1].split('.', maxsplit=1)[0]}.jpg"
            elif dataset_name == "vision_flan(filtered)": # replace '/' with '+'
                json_data["image"] = f"{dataset_name}/{da['id'].replace('/', '+')}.jpg"
            else:
                json_data["image"] = f"{dataset_name}/{da['id']}.jpg"
            if da["image"].mode != "RGB":
                da["image"].convert("RGB").save(os.path.join(image_folder, json_data["image"]))
            else:
                da["image"].save(os.path.join(image_folder, json_data["image"]))

        json_data["conversations"] = da["conversations"]
        converted_data.append(json_data)

    with open(os.path.join(json_path, json_file_name), "w") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)

    # count += 1
