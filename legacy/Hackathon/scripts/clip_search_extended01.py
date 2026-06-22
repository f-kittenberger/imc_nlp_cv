import json
import os
from collections import defaultdict

import torch

from src.model.clip_model import encode_text, encode_image


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

ATTRIBUTE_WORDS = {
    "red", "blue", "white", "black", "yellow", "green", "silver", "gray", "grey"
}


def parse_query(query: str) -> tuple[dict[str, int], list[str]]:
    tokens = query.lower().replace(",", " ").split()

    counts = {}
    attributes = []
    last_number = 1

    for token in tokens:
        if token in NUMBER_MAP:
            last_number = NUMBER_MAP[token]
            continue

        if token in ATTRIBUTE_WORDS:
            attributes.append(token)
            continue

        if token in CLASS_MAP:
            cls = CLASS_MAP[token]
            counts[cls] = last_number
            last_number = 1

    return counts, attributes


def load_crop_metadata(metadata_path: str) -> list[dict]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_index(crop_metadata: list[dict]) -> dict[str, dict]:
    image_index = {}

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


def filter_candidate_images(image_index: dict[str, dict], count_constraints: dict[str, int]) -> list[dict]:
    candidates = []

    for image_data in image_index.values():
        ok = True
        for cls_name, needed in count_constraints.items():
            if image_data["counts"].get(cls_name, 0) < needed:
                ok = False
                break

        if ok:
            candidates.append(image_data)

    return candidates


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b.T)


def score_image_for_query(
    image_data: dict,
    target_class: str,
    needed_count: int,
    query_text: str,
    threshold: float = 0.20,
) -> dict | None:
    relevant_crops = [
        crop for crop in image_data["crops"]
        if crop["class_name"] == target_class
    ]

    if len(relevant_crops) < needed_count:
        return None

    query_embedding = encode_text(query_text)
    crop_scores = []

    for crop in relevant_crops:
        crop_embedding = encode_image(crop["crop_path"])
        score = float(cosine_similarity(query_embedding, crop_embedding)[0, 0])
        crop_scores.append(
            {
                "crop_file": crop["crop_file"],
                "crop_path": crop["crop_path"],
                "score": score,
            }
        )

    crop_scores.sort(key=lambda x: x["score"], reverse=True)

    # Prüfe, ob mindestens needed_count Crops stark genug sind
    if len(crop_scores) < needed_count:
        return None

    if crop_scores[needed_count - 1]["score"] < threshold:
        return None

    return {
        "original_image": image_data["original_image"],
        "original_path": image_data["original_path"],
        "counts": dict(image_data["counts"]),
        "top_crops": crop_scores[:needed_count],
        "best_score": crop_scores[0]["score"],
        "min_required_score": crop_scores[needed_count - 1]["score"],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    metadata_path = os.path.join(project_root, "outputs", "crop_metadata.json")

    crop_metadata = load_crop_metadata(metadata_path)
    image_index = build_image_index(crop_metadata)

    while True:
        query = input("\nEnter structured query (or 'exit'): ").strip()
        if query.lower() == "exit":
            print("Exiting.")
            break

        if not query:
            print("Please enter a query.")
            continue

        count_constraints, attributes = parse_query(query)

        if not count_constraints:
            print("No supported object class found in query.")
            continue

        print(f"Parsed counts: {count_constraints}")
        print(f"Parsed attributes: {attributes}")

        candidate_images = filter_candidate_images(image_index, count_constraints)
        print(f"YOLO candidates: {len(candidate_images)}")

        # Erste Version: nur eine Zielklasse semantisch prüfen
        # Beispiel: "two red cars" -> target_class = car, needed_count = 2, query_text = "a red car"
        # Bei mehreren Klassen können wir im nächsten Schritt erweitern.
        if len(count_constraints) > 1:
            print("Current version supports CLIP reranking for one main class at a time.")
            print("YOLO filtering worked, but semantic reranking is currently single-class.")
            for image_data in candidate_images[:10]:
                print(f"-> {image_data['original_path']} | counts={dict(image_data['counts'])}")
            continue

        target_class, needed_count = next(iter(count_constraints.items()))

        if attributes:
            query_text = "a " + " ".join(attributes) + f" {target_class}"
        else:
            query_text = f"a {target_class}"

        print(f"CLIP reranking prompt: {query_text}")

        results = []
        for image_data in candidate_images:
            scored = score_image_for_query(
                image_data=image_data,
                target_class=target_class,
                needed_count=needed_count,
                query_text=query_text,
                threshold=0.20,
            )
            if scored is not None:
                results.append(scored)

        results.sort(key=lambda x: x["min_required_score"], reverse=True)

        print("\nTop results:")
        if not results:
            print("No result passed the CLIP threshold.")
            continue

        for result in results[:10]:
            print(f"\nImage: {result['original_path']}")
            print(f"Counts: {result['counts']}")
            print(f"Best score: {result['best_score']:.4f}")
            print(f"Required kth score: {result['min_required_score']:.4f}")
            for crop in result["top_crops"]:
                print(f"  {crop['score']:.4f} -> {crop['crop_path']}")


if __name__ == "__main__":
    main()