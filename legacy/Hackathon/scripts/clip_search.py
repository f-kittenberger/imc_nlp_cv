import os
import torch

from src.model.clip_model import encode_text
from src.retrieval.search import search


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    embeddings_path = os.path.join(project_root, "image_embeddings.pt")

    data = torch.load(embeddings_path, map_location=device)

    image_embeddings = data["embeddings"].to(device)
    image_paths = data["paths"]

    print("Embeddings loaded.")

    while True:
        query = input("\nEnter search text (or 'exit'): ").strip()

        if query.lower() == "exit":
            print("Exiting.")
            break

        if not query:
            print("Please enter a query.")
            continue

        query_embedding = encode_text(query).to(device)

        results = search(query_embedding, image_embeddings, image_paths, top_k=5)

        print("\nTop results:")
        for path, score in results:
            abs_path = os.path.abspath(path)
            print(f"{score:.4f} -> {abs_path}")


if __name__ == "__main__":
    main()