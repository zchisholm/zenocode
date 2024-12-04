import os
import tempfile
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from pinecone import Pinecone
from git import Repo
import json
from langchain.schema import Document

# Helper Functions

def clone_repo(repo_url):
    try:
        repo_path = tempfile.mkdtemp()
        Repo.clone_from(repo_url, repo_path)
        return repo_path
    except Exception as e:
        st.error(f"Failed to clone repository: {e}")
        return None


def get_file_content(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def get_main_files_content(repo_path):
    files_content = []
    for root, _, files in os.walk(repo_path):
        if any(ignored in root for ignored in IGNORED_DIRS):
            continue
        for file in files:
            if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                content = get_file_content(os.path.join(root, file))
                if content:
                    files_content.append({"name": file, "content": content})
    return files_content


def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)


def load_cached_embeddings(repo_path):
    cache_file = os.path.join(repo_path, "embeddings_cache.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


def save_embeddings_to_cache(repo_path, documents):
    cache_file = os.path.join(repo_path, "embeddings_cache.json")
    with open(cache_file, "w") as f:
        json.dump(documents, f)


@st.cache_resource
def initialize_pinecone():
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    return Pinecone(api_key=pinecone_api_key).Index(pinecone_index_name)

def perform_rag(query):
    query_embedding = get_huggingface_embeddings(query)
    results = pinecone_index.query(
        vector=query_embedding.tolist(), top_k=3, include_metadata=True, namespace=repo_link
    )
    contexts = [item["metadata"]["text"]
                for item in results.get("matches", [])]
    augmented_query = (
        "<CONTEXT>\n" + "\n\n---------\n\n".join(
            contexts[:10]) + "\n--------\n</CONTEXT>\n\nMY QUESTION:\n" + query
    )

    model = "llama-3.1-70b-versatile"

    with st.spinner("Generating a response..."):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": augmented_query}
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(
                "All models failed to generate a response. Please try again later.")


    return "Unable to generate a response at this time."


# Constants
SUPPORTED_EXTENSIONS = [".py", ".js", ".tsx", ".ts", ".java", ".cpp"]
IGNORED_DIRS = ["node_modules", ".git", "dist",
                "__pycache__", ".next", ".vscode", "env", "venv"]
SYSTEM_PROMPT = """
You are a Senior Software Engineer with over 20 years of experience in TypeScript.
Answer any questions about the codebase, using the provided context.
"""

# Streamlit Setup
st.title("Welcome to Zeno Code")
st.subheader("RAG over codebase")

# Clone Repository
repo_link = st.text_input("Enter GitHub Repo URL",
                          "https://github.com/zchisholm/ZensiReviews")
if repo_link:
    repo_path = clone_repo(repo_link)
    if not repo_path:
        st.stop()

# Fetch File Content
files = get_main_files_content(repo_path)
st.info(f"Loaded {len(files)} code files for analysis.")

# Pinecone Setup
pinecone_index_name = "zeno-code"
pinecone_index = initialize_pinecone()

# Insert Embeddings into Pinecone
documents = [Document(page_content=f"{file['name']}\n{file['content']}", metadata={
                      "source": file["name"]}) for file in files]
vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=HuggingFaceEmbeddings(),
    index_name=pinecone_index_name,
    namespace=repo_link,
)

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",  # Replace with your actual API URL
    api_key=st.secrets["GROQ_API_KEY"]
)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about the codebase..."):
    st.session_state["messages"].append({"role": "user", "content": query})
    response = perform_rag(query)
    st.session_state["messages"].append(
        {"role": "assistant", "content": response})
