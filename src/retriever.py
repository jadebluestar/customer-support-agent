"""
Load the retrieval corpus and embeddings; retrieve top-k by cosine similarity.

Both vectors are L2-normalized at build time, so cosine similarity reduces to
a dot product. A single numpy matrix multiply scores the whole corpus in
milliseconds — no vector DB needed at this scale.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.project_paths import repo_path

CORPUS_PATH = repo_path("data", "retrieval_corpus.csv")
EMB_PATH    = repo_path("data", "retrieval_embeddings.npy")
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievedCase:
    conversation_id: int
    customer_message: str
    brand_reply: str
    score: float


class Retriever:
    def __init__(self, corpus_path=CORPUS_PATH, emb_path=EMB_PATH):
        self.corpus = pd.read_csv(corpus_path)
        self.embeddings = np.load(emb_path)          # (N, 384), normalized
        self.model = SentenceTransformer(MODEL_NAME)
        assert len(self.corpus) == self.embeddings.shape[0], \
            "corpus and embeddings out of sync — re-run build_retrieval_corpus.py"

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedCase]:
        q = self.model.encode([query], convert_to_numpy=True,
                              normalize_embeddings=True).astype("float32")  # (1, 384)
        sims = (self.embeddings @ q.T).ravel()        # (N,)
        top = np.argsort(sims)[-k:][::-1]
        return [
            RetrievedCase(
                conversation_id=int(self.corpus.at[i, "conversation_id"]),
                customer_message=self.corpus.at[i, "customer_message"],
                brand_reply=self.corpus.at[i, "brand_reply"],
                score=float(sims[i]),
            )
            for i in top
        ]


if __name__ == "__main__":
    r = Retriever()
    for q in [
        "my package was supposed to arrive 3 days ago and it's still not here",
        "why was I charged twice this month",
        "how do I return an item",
    ]:
        print(f"\nQuery: {q}")
        for hit in r.retrieve(q, k=3):
            print(f"  [{hit.score:.3f}] C: {hit.customer_message[:80]}")
            print(f"          R: {hit.brand_reply[:80]}")