from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#We are using a small sample dataset to simulate semantic searching
documents = [

    "Python is a programming language",

    "Football is a popular sport",

    "Machine learning is part of AI",

    "Dogs are loyal animals",

    "Cats like sleeping"

]
#Loading the embedding model
model=SentenceTransformer('all-MiniLM-L6-v2')

#Convert text to embeddings
doc_embeddings=model.encode(documents)

doc_embeddings=np.array(doc_embeddings)

dimension=doc_embeddings.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)


# User query

query = "What is artificial intelligence?"

# Convert query into embedding

query_embedding = model.encode([query])
k=2
distances, indices = index.search(

    np.array(query_embedding),

    k

)

print("\nTop Matches:\n")

for idx in indices[0]:

    print(documents[idx])