import os
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to extract text from a Word document
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Path to the document (can be either a single file or a directory)
docs_path = "C:\\Users\\MOD\\chatbot\\Should companies implement a four.docx"

documents = []
doc_texts = []

if os.path.isdir(docs_path):
    # Iterate through all files in the directory
    for filename in os.listdir(docs_path):
        file_path = os.path.join(docs_path, filename)
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
            documents.append(filename)
            doc_texts.append(text)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(file_path)
            documents.append(filename)
            doc_texts.append(text)
elif os.path.isfile(docs_path):
    # Process a single file
    if docs_path.endswith(".pdf"):
        text = extract_text_from_pdf(docs_path)
        documents.append(os.path.basename(docs_path))
        doc_texts.append(text)
    elif docs_path.endswith(".docx"):
        text = extract_text_from_docx(docs_path)
        documents.append(os.path.basename(docs_path))
        doc_texts.append(text)
else:
    print("Invalid path specified. Please provide a valid file or directory path.")

# Generate embeddings for the document texts
embeddings = embedding_model.encode(doc_texts)

# Create a FAISS index
d = embeddings.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(d)  # L2 distance metric
index.add(np.array(embeddings))  # Add embeddings to the index

# Save the FAISS index and metadata
index_path = "faiss_index"
if not os.path.exists(index_path):
    os.makedirs(index_path)

faiss.write_index(index, os.path.join(index_path, "index.faiss"))

# Save the document metadata to a file for retrieval purposes
with open(os.path.join(index_path, "documents.txt"), "w") as f:
    for doc in documents:
        f.write("%s\n" % doc)

# Save additional metadata
metadata = {
    "documents": documents,
    "embeddings": embeddings
}
with open(os.path.join(index_path, "index.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print("FAISS index and documents saved.")

# Load the FAISS index and metadata
index = faiss.read_index(os.path.join(index_path, "index.faiss"))
with open(os.path.join(index_path, "index.pkl"), "rb") as f:
    metadata = pickle.load(f)
documents = metadata["documents"]
embeddings = metadata["embeddings"]

# Retrieve the API token from the environment variable
api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
if api_token is None:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is not set")

print(f"API Token: {api_token[:5]}...")  # Print the first 5 characters of the token for verification

# Initialize the LLM
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/gpt2",
    model_kwargs={"api_key": api_token}
)

# Function to perform a search query
def search(query, k=5):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    results = [documents[i] for i in I[0]]
    return results

# Example query
query = "What is the impact of a four-day work week?"
results = search(query)
print("Top documents:", results)