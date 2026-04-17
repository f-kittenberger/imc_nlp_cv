import json
import os
from collections import defaultdict

import torch

from src.model.clip_model import encode_image, encode_text


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


def parse_requested_classes(query: str) -> list[str]:
    tokens = query.lower().replace(",", " ").split()
    found_classes = []

    for token in tokens:
        if token in CLASS_MAP:
            cls = CLASS_MAP[token]
            if cls not in found_classes:
                found_classes.append(cls)

    return found_classes


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
    requested_classes: list[str],
) -> list[dict]:
    candidates = []

    for image_data in image_index.values():
        counts = image_data["counts"]

        ok = True
        for cls_name in requested_classes:
            if counts.get(cls_name, 0) < 1:
                ok = False
                break

        if ok:
            candidates.append(image_data)

    return candidates


def build_yolo_summary(counts: dict) -> str:
    car_count = counts.get("car", 0)
    truck_count = counts.get("truck", 0)
    bus_count = counts.get("bus", 0)
    motorcycle_count = counts.get("motorcycle", 0)

    return (
        f"detected objects: "
        f"{car_count} cars, "
        f"{truck_count} trucks, "
        f"{bus_count} buses, "
        f"{motorcycle_count} motorcycles"
    )


def score_candidate_images(
    candidate_images: list[dict],
    full_query: str,
) -> list[dict]:
    query_embedding = encode_text(full_query)
    results = []

    for image_data in candidate_images:
        image_embedding = encode_image(image_data["original_path"])

        # Score 1: nur die originale User-Query
        query_score = float(cosine_similarity(query_embedding, image_embedding)[0, 0])

        # Score 2: User-Query + YOLO-Zusatzinfo
        yolo_summary = build_yolo_summary(image_data["counts"])
        augmented_query = f"{full_query}. {yolo_summary}"
        augmented_query_embedding = encode_text(augmented_query)

        augmented_score = float(
            cosine_similarity(augmented_query_embedding, image_embedding)[0, 0]
        )

        # Finale Kombination
        final_score = 0.7 * query_score + 0.3 * augmented_score

        results.append(
            {
                "original_image": image_data["original_image"],
                "original_path": image_data["original_path"],
                "counts": dict(image_data["counts"]),
                "yolo_summary": yolo_summary,
                "query_score": query_score,
                "augmented_score": augmented_score,
                "final_score": final_score,
            }
        )

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    metadata_path = os.path.join(project_root, "outputs", "crop_metadata_OR.json")

    crop_metadata = load_crop_metadata(metadata_path)
    image_index = build_image_index(crop_metadata)

    print(f"Loaded metadata for {len(image_index)} original images.")

    while True:
        query = input("\nEnter query (or 'exit'): ").strip()

        if query.lower() == "exit":
            print("Exiting.")
            break

        if not query:
            print("Please enter a query.")
            continue

        requested_classes = parse_requested_classes(query)

        if not requested_classes:
            print("No supported object class found in query.")
            print("Supported classes: car, bus, truck, motorcycle")
            continue

        print(f"Requested classes: {requested_classes}")

        candidate_images = filter_candidate_images(image_index, requested_classes)
        print(f"YOLO candidates: {len(candidate_images)}")

        if not candidate_images:
            print("No images satisfy the requested object classes.")
            continue

        ranked_results = score_candidate_images(candidate_images, query)

        print("\nTop results:")

        for result in ranked_results[:4]:
            print("\n" + "=" * 60)
            print(f"Final score:     {result['final_score']:.4f}")
            print(f"Query score:     {result['query_score']:.4f}")
            print(f"Augmented score: {result['augmented_score']:.4f}")
            print(f"Image:           {result['original_path']}")
            print(f"Counts:          {result['counts']}")
            print(f"YOLO info:       {result['yolo_summary']}")

            image_name = result["original_image"]
            image_data = image_index[image_name]

            crops_by_class = defaultdict(list)
            for crop in image_data["crops"]:
                crops_by_class[crop["class_name"]].append(crop["crop_path"])

            print("Crops:")
            for cls_name, crop_list in crops_by_class.items():
                print(f"    {cls_name}:")
                for crop_file in crop_list:
                    print(f"        - {crop_file}")


if __name__ == "__main__":
    main()