from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

reader=PdfReader('AgriScribe-Plagiarism-Report.pdf')
model=SentenceTransformer('all-MiniLM-L6-v2')

api_key=os.getenv("MY-API-KEY")
genai.configure(api_key=api_key)

#Loading the model
model_llm=genai.GenerativeModel("gemini-3.5-flash")

text=""
for page in reader.pages:
    text+=page.extract_text()

chunk_size=200
overlap=50
chunks=[]

for i in range (0,len(text),chunk_size-overlap):
    chunk=text[i:i+chunk_size]
    chunks.append(chunk)

# print(f"Total chunks: {len(chunks)}")
# print(f"\nFirst Chunk\n")
# print(chunks[0])


embeddings=model.encode(chunks)
# print(embeddings.shape)
dimension=embeddings.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(embeddings)


query="What is the project about?"

query_embedding=model.encode([query])
k = 1

distances, indices = index.search(
    np.array(query_embedding),
    k
)
# print(indices)
# print("\nBest Matching Chunk:\n")

# print(chunks[indices[0][0]])
retrieved_chunks=[]
for idx in indices[0]:
    retrieved_chunks.append(chunks[idx])

context = "\n".join(retrieved_chunks)
prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

response = model_llm.generate_content(prompt)
print("\nANSWER:\n")

print(response.text)