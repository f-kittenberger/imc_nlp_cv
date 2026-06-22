import json
import os
from collections import defaultdict
from unittest import result

import torch

from src.model.clip_model import encode_image, encode_text


NUMBER_MAP = {
    "a": 1,
    "an": 1,
    "one": 1,
    "1": 1,
    "two": 2,
    "2": 2,
    "three": 3,
    "3": 3,
    "four": 4,
    "4": 4,
    "five": 5,
    "5": 5,
    "multiple": 2,
    "many": 3,
}

CLASS_MAP = {
    "car": "car",
    "cars": "car",
    "bus": "bus",
    "buses": "bus",
    "truck": "truck",
    "trucks": "truck",
    "motorcycle": "motorcycle",
    "motorcycles": "motorcycle",
    "bike": "motorcycle",
    "bikes": "motorcycle",
}


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b.T)


def parse_count_constraints(query: str) -> dict[str, int]:
    tokens = query.lower().replace(",", " ").split()

    counts: dict[str, int] = {}
    last_number = 1

    for token in tokens:
        if token in NUMBER_MAP:
            last_number = NUMBER_MAP[token]
            continue

        if token in CLASS_MAP:
            cls = CLASS_MAP[token]
            counts[cls] = last_number
            last_number = 1

    return counts


def load_crop_metadata(metadata_path: str) -> list[dict]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_index(crop_metadata: list[dict]) -> dict[str, dict]:
    image_index: dict[str, dict] = {}

    for item in crop_metadata:
        original_image = item["original_image"]

        if original_image not in image_index:
            image_index[original_image] = {
                "original_image": original_image,
                "original_path": item["original_path"],
                "counts": defaultdict(int),
                "crops": [],
            }

        cls_name = item["class_name"]
        image_index[original_image]["counts"][cls_name] += 1
        image_index[original_image]["crops"].append(item)

    return image_index


def filter_candidate_images(
    image_index: dict[str, dict],
    count_constraints: dict[str, int],
    exact_match: bool = True,
) -> list[dict]:
    candidates = []

    all_classes = ["car", "bus", "truck", "motorcycle"]

    for image_data in image_index.values():
        counts = image_data["counts"]
        ok = True

        if exact_match:
            # genannte Klassen muessen exakt stimmen
            for cls_name, needed_count in count_constraints.items():
                if counts.get(cls_name, 0) != needed_count:
                    ok = False
                    break

            # nicht genannte Klassen muessen 0 sein
            if ok:
                for cls_name in all_classes:
                    if cls_name not in count_constraints and counts.get(cls_name, 0) != 0:
                        ok = False
                        break
        else:
            # alte Logik: mindestens so viele
            for cls_name, needed_count in count_constraints.items():
                if counts.get(cls_name, 0) < needed_count:
                    ok = False
                    break

        if ok:
            candidates.append(image_data)

    return candidates


def score_candidate_images(
    candidate_images: list[dict],
    full_query: str,
) -> list[dict]:
    query_embedding = encode_text(full_query)
    results = []

    for image_data in candidate_images:
        image_embedding = encode_image(image_data["original_path"])
        score = float(cosine_similarity(query_embedding, image_embedding)[0, 0])

        results.append(
            {
                "original_image": image_data["original_image"],
                "original_path": image_data["original_path"],
                "counts": dict(image_data["counts"]),
                "score": score,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    metadata_path = os.path.join(project_root, "outputs", "crop_metadata.json")

    crop_metadata = load_crop_metadata(metadata_path)
    image_index = build_image_index(crop_metadata)

    print(f"Loaded metadata for {len(image_index)} original images.")

    while True:
        query = input("\nEnter structured query (or 'exit'): ").strip()

        if query.lower() == "exit":
            print("Exiting.")
            break

        if not query:
            print("Please enter a query.")
            continue

        count_constraints = parse_count_constraints(query)

        if not count_constraints:
            print("No supported object class found in query.")
            print("Supported classes: car, bus, truck, motorcycle")
            continue

        print(f"Parsed counts: {count_constraints}")

        candidate_images = filter_candidate_images(image_index, count_constraints)
        print(f"YOLO candidates: {len(candidate_images)}")

        if not candidate_images:
            print("No images satisfy the YOLO count constraints.")
            continue

        print(f"CLIP reranking with full query: {query}")

        ranked_results = score_candidate_images(candidate_images, query)


        print("\nTop results:")

        for result in ranked_results[:4]:
            print("\n" + "=" * 60)
            print(f"Score: {result['score']:.4f}")
            print(f"Image: {result['original_path']}")
            print(f"Counts: {result['counts']}")

            image_name = result["original_image"]
            image_data = image_index[image_name]

            crops_by_class = defaultdict(list)
            for crop in image_data["crops"]:
                crops_by_class[crop["class_name"]].append(crop["crop_file"])

            print("Crops:")

            for cls_name, crop_list in crops_by_class.items():
                print(f"    {cls_name}:")
                for crop_file in crop_list:
                    print(f"        - {crop_file}")

if __name__ == "__main__":
    main()