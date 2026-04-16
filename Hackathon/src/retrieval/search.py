import torch


def search(query_embedding, image_embeddings, image_paths, top_k=5):
    similarities = torch.matmul(query_embedding, image_embeddings.T)
    top_indices = similarities[0].topk(top_k).indices.tolist()

    results = []
    for idx in top_indices:
        results.append((image_paths[idx], float(similarities[0, idx])))

    return results