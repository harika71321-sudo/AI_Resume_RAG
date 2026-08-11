import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, embedding_model: str):
        self.embedding_model_name = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.metadata = []

    def build(self, documents: list[dict]):
        texts = [doc["text"] for doc in documents]

        if not texts:
            raise ValueError("No text chunks were supplied.")

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.metadata = documents

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        if self.index is None or not self.metadata:
            raise ValueError("Vector store is empty.")

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        query_embedding = np.asarray(query_embedding, dtype="float32")

        k = min(top_k, len(self.metadata))
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)

        return results

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.index is None:
            raise ValueError("Nothing to save.")

        faiss.write_index(self.index, str(directory / "resume.index"))

        with open(directory / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, directory: str | Path):
        directory = Path(directory)

        index_path = directory / "resume.index"
        metadata_path = directory / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Saved vector store not found.")

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def clear(self):
        self.index = None
        self.metadata = []
