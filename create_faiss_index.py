import os
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

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