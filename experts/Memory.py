import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Use a small, efficient embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample knowledge database (text corpus)
knowledge_base = [
    "The Eiffel Tower is in Paris.",
    "Albert Einstein developed the theory of relativity.",
    "The capital of Japan is Tokyo."
]

# Convert knowledge into embeddings
kb_embeddings = np.array(embed_model.encode(knowledge_base)).astype("float32")

# Initialize FAISS index (Flat L2 search)
index = faiss.IndexFlatL2(kb_embeddings.shape[1])
index.add(kb_embeddings)

# Query: "Where is the Eiffel Tower?"
query = "Where is the Eiffel Tower located?"
query_embedding = np.array(embed_model.encode([query])).astype("float32")

# Search in the knowledge base
D, I = index.search(query_embedding, k=1)  # k=1 returns the closest match

print("Retrieved knowledge:", knowledge_base[I[0][0]])
