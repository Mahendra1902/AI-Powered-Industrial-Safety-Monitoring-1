# vector_db.py

import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
from vector_db # type: ignore
import numpy as np

class FAISSVectorDB:
    def __init__(self, index_path="faiss_index", model_name='all-MiniLM-L6-v2'):
        self.index_path = index_path
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.data = []

    def load_index(self):
        if os.path.exists(self.index_path + ".index"):
            self.index = faiss.read_index(self.index_path + ".index")
            with open(self.index_path + "_data.pkl", "rb") as f:
                self.data = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
            self.data = []

    def save_index(self):
        faiss.write_index(self.index, self.index_path + ".index")
        with open(self.index_path + "_data.pkl", "wb") as f:
            pickle.dump(self.data, f)

    def add_documents(self, documents):
        embeddings = self.model.encode(documents)
        self.index.add(np.array(embeddings))
        self.data.extend(documents)
        self.save_index()

    def search(self, query, top_k=3):
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vec), top_k)
        return [self.data[i] for i in indices[0]]
