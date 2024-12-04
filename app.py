from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
import os
import tempfile
from github import Github, Repository
from git import Repo
from openai import OpenAI
from pathlib import Path
from langchain.schema import Document
from pinecone import Pinecone
import tempfile
import streamlit as st

# Cloning the Repo
def clone_repo(repo_url):
    repo_name = repo_url.split("/")[-1]
    repo_path = tempfile.mkdtemp()
    Repo.clone_from(repo_url, str(repo_path))
    return str(repo_name)

# Obtain Content
def get_file_content(file_path, repo_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        rel_path = os.path.relpath(file_path, repo_path)

        return {
            "name": rel_path,
            "content": content
        }
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

# Main Files    
def get_main_files_content(repo_path: str):
   """
   Get content of supported code files from the local repository.


   Args:
       repo_path: Path to the local repository


   Returns:
       List of dictionaries containing file names and contents
   """
   files_content = []


   try:
       for root, _, files in os.walk(repo_path):
           # Skip if current directory is in ignored directories
           if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
               continue


           # Process each file in current directory
           for file in files:
               file_path = os.path.join(root, file)
               if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                   file_content = get_file_content(file_path, repo_path)
                   if file_content:
                       files_content.append(file_content)


   except Exception as e:
       print(f"Error reading repository: {str(e)}")


   return files_content

# Embeddings
def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# Perform RAG get response from LLM
def perform_rag(query):
    query_embedding = get_huggingface_embeddings(query)
    top_matches = pinecone_index.query(vector=query_embedding.tolist(), top_k=5,
                                       include_metadata=True, namespace=repo_link)
    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    augmented_query = "<CONTEXT>\n" + "\n\n---------\n\n".join(
        contexts[:10]) + "\n--------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
    
    system_prompt = f"""You are a Senior Software Engineer, with over 20 years of experience in TypeScript.

    Answer any questions I have about the codebase, based on all the context provided.
    Always consider all of the context provided when forming a response.

    Let's think step by step
    """


    llm_response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    response = llm_response.choices[0].message.content

    return response


# Set
repo_link = 'https://github.com/zchisholm/ZensiReviews'

path = clone_repo(repo_link)

SUPPORTED_EXTENSIONS = [".py", ".js", ".tsx", ".ts", ".java", ".cpp"]

IGNORED_DIRS = ["node_modules", ".git", "dist",
                "__pycache__", ".next", ".vscode", "env", "venv"]


file_content = get_main_files_content(path)

# Set the PINECONE_API_KEY as an environment variable
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
os.environ['PINECONE_API_KEY'] = pinecone_api_key

# Initialize Pinecone
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

# Connect to your Pinecone index
pc_indexname = "zeno-code"
pinecone_index = pc.Index(pc_indexname)


vectorstore = PineconeVectorStore(
    index_name=pc_indexname, embedding=HuggingFaceEmbeddings())


# Insert the codebase embeddings into Pinecone

documents = []

for file in file_content:
    doc = Document(
        page_content=f"{file['name']}\n{file['content']}",
        metadata={"source": file['name']}
    )
    documents.append(doc)

vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=HuggingFaceEmbeddings(),
    index_name=pc_indexname,
    namespace=repo_link
)


st.title("Zeno Code - RAG over Codebase")

client = OpenAI(base_url="https://api.grow.com/openaiv1",
                api_key=st.secrets["GROQ_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "llama-31.1-70b-versatile"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything about the codebase..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = perform_rag(prompt)

    st.session_state.messages.append(
        {"role": "assistant", "content": response})
