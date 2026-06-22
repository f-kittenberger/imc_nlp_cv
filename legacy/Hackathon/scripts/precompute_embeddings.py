import os
import torch

from src.model.clip_model import encode_image
from src.data.loader import load_dataset


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json_path = os.path.join(project_root, "vehicle_subset_descriptions.json")

    image_paths, _ = load_dataset(json_path, project_root)

    print(f"Encoding {len(image_paths)} images...")

    image_embeddings = []

    for i, path in enumerate(image_paths):
        print(f"{i+1}/{len(image_paths)}")
        emb = encode_image(path)
        image_embeddings.append(emb)

    image_embeddings = torch.cat(image_embeddings, dim=0)

    # speichern
    torch.save({
        "embeddings": image_embeddings,
        "paths": image_paths
    }, os.path.join(project_root, "image_embeddings.pt"))

    print("Saved embeddings!")


if __name__ == "__main__":
    main()