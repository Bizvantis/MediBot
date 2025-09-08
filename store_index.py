import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec # [2, 3] - Pinecone client for index management
from langchain_pinecone import PineconeVectorStore # [1, 4] - For interacting with Pinecone via Langchain
from SRC.helper import load_PDF, text_split, download_hugging_face_embedding_model# [1] - Custom utility functions from src/helper.py

# Load environment variables from .env file [2, 3]
load_dotenv()

# Set Pinecone API key from environment variables [2, 3]
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") # [3]
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY # [4]

# Define constants for the Pinecone index [2, 3]
INDEX_NAME = "medical-bot" # [2, 3]
EMBEDDING_DIMENSION = 384 # The dimension of the vector embeddings, as stated for the 'all-MiniLM-L6-v2' model [5, 6]
CLOUD_PROVIDER = "aws" # Free tier uses AWS [6]
AWS_REGION = "us-east-1" # Default region for the free tier [3, 6]
DATA_DIRECTORY = "data/" # Location where the PDF medical book is stored [2, 7]

def store_embeddings_in_pinecone():
    """
    Orchestrates the process of loading medical data,
    splitting it into chunks, generating embeddings,
    and storing them in Pinecone.
    """
    print("Starting the embedding storage process...")

    # Load data from PDF documents [2]
    print(f"Loading PDF data from {DATA_DIRECTORY}...")
    extracted_data = load_PDF(DATA_DIRECTORY) # [2]
    print(f"Extracted data from {len(extracted_data)} pages.")

    # Split the extracted data into smaller chunks [2]
    print("Splitting data into text chunks...")
    text_chunks = text_split(extracted_data) # [2]
    print(f"Created {len(text_chunks)} text chunks.")

    # Download the embedding model from Hugging Face [2]
    print("Downloading Hugging Face embedding model...")
    embeddings = download_hugging_face_embedding_model() # [2]
    print("Embedding model downloaded successfully.")

    # Initialize Pinecone [2]
    pinecone = Pinecone(api_key=PINECONE_API_KEY) # [2, 3]

    # Check if the index already exists in Pinecone, if not, create it [3, 6]
    if INDEX_NAME not in pinecone.list_indexes():
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pinecone.create_index(
            name=INDEX_NAME, # [3]
            dimension=EMBEDDING_DIMENSION, # [3]
            metric="cosine", # Metric used for similarity search [6]
            spec=ServerlessSpec(cloud=CLOUD_PROVIDER, region=AWS_REGION) # [3]
        )
        print(f"Pinecone index '{INDEX_NAME}' created.")
    else:
        print(f"Pinecone index '{INDEX_NAME}' already exists.")

    # Store the text chunks and their embeddings into the Pinecone index [2, 4]
    print("Storing embeddings in Pinecone. This may take some time...")
    PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=INDEX_NAME) # [2, 4]
    print("Embeddings successfully stored in Pinecone.")
    print("Process completed. Your knowledge base is ready.")

if __name__ == "__main__":
    store_embeddings_in_pinecone()