import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

load_dotenv()

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("RAG Document Assistant")

client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL")
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                collection = chroma_client.get_or_create_collection(name="documents")
                reader = PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
                
                chunk_size = 500
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

                doc_key = uploaded_file.name.replace(" ", "_")
                for i, chunk in enumerate(chunks):
                    emb = model.encode(chunk).tolist()
                    collection.upsert(
                        embeddings=[emb],
                        documents=[chunk],
                        metadatas=[{"source": uploaded_file.name, "chunk": i}],
                        ids=[f"{doc_key}_{i}"]
                    )
                st.success(f"Added {len(chunks)} chunks to database.")

    if st.button("Clear database"):
        chroma_client.delete_collection(name="documents")
        st.session_state.messages = []
        st.success("Database cleared.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the document"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        collection = chroma_client.get_collection(name="documents")
        query_emb = model.encode(prompt).tolist()
        
        try:
            results = collection.query(query_embeddings=[query_emb], n_results=3)
            context = "\n".join(results['documents'][0])
            
            full_prompt = f"""Use the following context to answer the question. 
If you don't know the answer, say that you don't know.

Context:
{context}

Question: {prompt}
Answer:"""
            
            response = client.chat.completions.create(
                model=os.getenv("QWEN_MODEL", "qwen-plus"),
                messages=[{"role": "user", "content": full_prompt}]
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"API Error: {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})