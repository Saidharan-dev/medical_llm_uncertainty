import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_variance(texts):
    embeddings = embedder.encode(texts)
    n = len(embeddings)

    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(cosine(embeddings[i], embeddings[j]))

    return float(np.mean(distances)) if distances else 0.0
