import json
import os


def load_dataset(json_path, project_root):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_paths = []
    all_descriptions = []

    for item in data:
        image_path = os.path.join(project_root, item["image_path"])
        descriptions = item["descriptions"]

        if os.path.exists(image_path):
            image_paths.append(image_path)
            all_descriptions.append(descriptions)

    return image_paths, all_descriptions