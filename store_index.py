
# runned this once and vector db created

from SRC.helper import load_pdf_data, get_text_chunks, download_hugging_face_embeddings # Import utility functions from helper.py
from pinecone import Pinecone, ServerlessSpec # For Pinecone interaction and serverless specification
from langchain_pinecone import PineconeVectorStore # LangChain integration with Pinecone
from dotenv import load_dotenv # For loading environment variables from .env file
import os # For interacting with the operating system, particularly for environment variables

# Load all environment variables from the .env file
load_dotenv()

# Retrieve Pinecone API key from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Although not directly used for Pinecone initialisation, the OpenAI API key is also typically loaded here if needed elsewhere
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

# Set Pinecone API key in the environment for LangChain's internal use [3]
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Define the directory where your PDF data is stored [2, 4]
DATA_DIRECTORY = "data/"
# Define the name for your Pinecone index [2, 5]
INDEX_NAME = "medical-chatbot" 

if __name__ == "__main__":
    print("Starting data ingestion and index storage process...")

    # 1. Load data from PDF files located in the specified directory [2, 6]
    extracted_data = load_pdf_data(DATA_DIRECTORY)
    print(f"Number of pages extracted: {len(extracted_data)}")

    # 2. Split the extracted data into manageable text chunks [2, 6]
    text_chunks = get_text_chunks(extracted_data)
    print(f"Number of text chunks created: {len(text_chunks)}")

    # 3. Download the Hugging Face embedding model [2, 7]
    embeddings = download_hugging_face_embeddings()
    print("Hugging Face embedding model downloaded.")

    # 4. Initialise Pinecone and create the index if it doesn't already exist [2, 5]
    # The environment "us-east-1" is an example; choose the region closest to you or as provided by Pinecone [5]
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1") 
    print("Pinecone client initialised.")

    # Check if the index exists before attempting to create it [5]
    if INDEX_NAME not in pinecone_client.list_indexes():
        print(f"Pinecone index '{INDEX_NAME}' does not exist. Creating it now...")
        # Create the index with the appropriate dimension (384 for 'all-MiniLM-L6-v2') [5, 7]
        pinecone_client.create_index(
            name=INDEX_NAME,
            dimension=384, # Dimension based on the 'all-MiniLM-L6-v2' model [7]
            metric='cosine', # Similarity metric for vector search [8]
            spec=ServerlessSpec(cloud='aws', region='us-east-1') # Serverless instance on AWS (example) [8]
        )
        print(f"Pinecone index '{INDEX_NAME}' created successfully.")
    else:
        print(f"Pinecone index '{INDEX_NAME}' already exists. Skipping creation.")

    # 5. Store the text chunks (converted to vector embeddings) in the Pinecone index [2, 3]
    print("Storing embeddings in Pinecone. This may take some time...")
    PineconeVectorStore.from_documents(
        text_chunks, # The documents to be embedded and stored
        embeddings, # The embedding model to use
        index_name=INDEX_NAME # The name of the Pinecone index
    )
    print("All embeddings stored in Pinecone.")
    print("Data ingestion and index storage process completed.")