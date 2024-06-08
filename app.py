import os
import fitz
from docx import Document
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import faiss
import numpy as np
import pickle
import gradio as gr
from typing import List
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

# Function to extract text from a Word document
def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
    return text

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Hugging Face API token
api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
if not api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is not set")

# Initialize RAG models from Hugging Face
generator_model_name = "facebook/bart-base"
retriever_model_name = "facebook/bart-base"
generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
retriever = AutoModelForSeq2SeqLM.from_pretrained(retriever_model_name)
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)    

# Initialize the HuggingFace LLM
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/gpt2",
    model_kwargs={"api_key": api_token}
)

# Initialize the HuggingFace embeddings
embedding = HuggingFaceEmbeddings()

# Load or create FAISS index
index_path = "faiss_index.pkl"
document_texts_path = "document_texts.pkl"

document_texts = []

if os.path.exists(index_path) and os.path.exists(document_texts_path):
    try:
        with open(index_path, "rb") as f:
            index = pickle.load(f)
            print("Loaded FAISS index from faiss_index.pkl")
        with open(document_texts_path, "rb") as f:
            document_texts = pickle.load(f)
            print("Loaded document texts from document_texts.pkl")
    except Exception as e:
        print(f"Error loading FAISS index or document texts: {e}")
else:
    # Create a new FAISS index if it doesn't exist
    index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    with open(index_path, "wb") as f:
        pickle.dump(index, f)
        print("Created new FAISS index and saved to faiss_index.pkl")

def upload_files(files):
    global index, document_texts
    try:
        for file in files:
            file_path = file.name  # Get the file path from the NamedString object
            if file_path.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            else:
                return "Unsupported file format"

            print(f"Extracted text: {text[:100]}...")  # Debug: Show the first 100 characters of the extracted text

            # Process the text and update FAISS index
            sentences = text.split("\n")
            embeddings = embedding_model.encode(sentences)
            print(f"Embeddings shape: {embeddings.shape}")  # Debug: Show the shape of the embeddings
            index.add(np.array(embeddings))
            document_texts.extend(sentences)  # Store sentences for retrieval

        # Save the updated index and documents
        with open(index_path, "wb") as f:
            pickle.dump(index, f)
            print("Saved updated FAISS index to faiss_index.pkl")
        with open(document_texts_path, "wb") as f:
            pickle.dump(document_texts, f)
            print("Saved updated document texts to document_texts.pkl")
        
        return "Files processed successfully"
    except Exception as e:
        print(f"Error processing files: {e}")
        return f"Error processing files: {e}"

def query_text(text):
    try:
        print(f"Query text: {text}")  # Debug: Show the query text

        # Encode the query text
        query_embedding = embedding_model.encode([text])
        print(f"Query embedding shape: {query_embedding.shape}")  # Debug: Show the shape of the query embedding
        
        # Search the FAISS index
        D, I = index.search(np.array(query_embedding), k=5)
        print(f"Distances: {D}, Indices: {I}")  # Debug: Show the distances and indices of the search results
        
        top_documents = []
        for idx in I[0]:
            if idx != -1 and idx < len(document_texts):  # Ensure that a valid index is found
                top_documents.append(document_texts[idx])  # Append the actual sentences for the response
            else:
                print(f"Invalid index found: {idx}")
        return top_documents
    except Exception as e:
        print(f"Error querying text: {e}")
        return f"Error querying text: {e}"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Document Upload and Query System")
    
    with gr.Tab("Upload Files"):
        upload = gr.File(file_count="multiple", label="Upload PDF or DOCX files")
        upload_button = gr.Button("Upload")
        upload_output = gr.Textbox()
        upload_button.click(fn=upload_files, inputs=upload, outputs=upload_output)
    
    with gr.Tab("Query"):
        query = gr.Textbox(label="Enter your query")
        query_button = gr.Button("Search")
        query_output = gr.Textbox()
        query_button.click(fn=query_text, inputs=query, outputs=query_output)

demo.launch()

