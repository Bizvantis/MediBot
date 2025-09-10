from langchain.document_loaders import PyPDFLoader, DirectoryLoader # For loading PDF documents [1, 4]
from langchain.text_splitter import RecursiveCharacterTextSplitter # For splitting text into chunks [1, 4]
from langchain.embeddings import HuggingFaceEmbeddings # For downloading and using embedding models [1, 5]

# Function to extract data from PDF files [1, 4]
def load_pdf_data(data_directory):
    """
    Loads PDF documents from a specified directory.
    
    Args:
        data_directory (str): The path to the directory containing PDF files.
        
    Returns:
        list: A list of loaded documents.
    """
    loader = DirectoryLoader(data_directory,
                             glob="*.pdf", # Only load PDF documents [4]
                             loader_cls=PyPDFLoader) # Uses PyPDFLoader to extract information [4]
    documents = loader.load()
    return documents

# Function to split extracted data into text chunks [1, 4]
def get_text_chunks(extracted_data):
    """
    Splits the extracted document data into smaller text chunks.
    
    Args:
        extracted_data (list): A list of documents extracted from PDFs.
        
    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, # Defines a chunk size of 500 characters [4]
                                                   chunk_overlap=20) # Defines a chunk overlap of 20 characters [4]
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Function to download the Hugging Face embedding model [1, 5]
def download_hugging_face_embeddings():
    """
    Downloads and initialises a Hugging Face embedding model.
    
    Returns:
        HuggingFaceEmbeddings: The loaded embedding model.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Uses the 'all-MiniLM-L6-v2' model [5]
    return embeddings
