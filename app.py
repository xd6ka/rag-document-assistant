import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")
model = SentenceTransformer('all-MiniLM-L6-v2')

def process_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
            
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        emb = model.encode(chunk).tolist()
        collection.add(
            embeddings=[emb],
            documents=[chunk],
            ids=[f"doc_{i}"]
        )
    print(f"Ingestion complete. Added {len(chunks)} chunks to the database.")

if __name__ == "__main__":
    process_pdf("data/real.pdf")