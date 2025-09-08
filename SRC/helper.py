from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_PDF(data_directory):
    """
    Loads all PDF documents from the specified data directory.

    Args:
        data_directory (str): The path to the directory containing PDF files.

    Returns:
        list: A list of Document objects extracted from the PDFs.
    """
    loader = DirectoryLoader(data_directory, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    """
    Splits the extracted data into smaller, manageable text chunks.

    Args:
        extracted_data (list): A list of Document objects from which to create chunks.

    Returns:
        list: A list of text chunks (Document objects).
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def downloadFaceEmbeddings():
    """
    Downloads and initializes a Hugging Face embedding model.

    Returns:
        HuggingFaceEmbeddings: An initialized Hugging Face embedding model.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model