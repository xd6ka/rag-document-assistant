import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL")
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="documents")
model = SentenceTransformer('all-MiniLM-L6-v2')

def ask(question):
    query_emb = model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=3)
    
    context = "\n".join(results['documents'][0])
    
    prompt = f"""Use the following context to answer the question. 
If you don't know the answer, say that you don't know.

Context:
{context}

Question: {question}
Answer:"""
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"API Error. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return f"API Error: {e}"

if __name__ == "__main__":
    print("RAG Assistant initialized. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
        answer = ask(user_input)
        print(f"AI: {answer}")