import os
from dotenv import load_dotenv

# Flask for web application interface
from flask import Flask, request, render_template, jsonify

# Custom helper functions
from SRC.helper import download_hugging_face_embedding_model, detect_language, preprocess_hindi_text

# Pinecone Vector Store for knowledge base interaction
from langchain_pinecone import PineconeVectorStore

# OpenAI Large Language Model
from langchain_groq import ChatGroq

# LangChain components for building the RAG chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Import the system prompts from src/prompt.py
from SRC.prompt import system_prompt, system_prompt_en, system_prompt_hi

# --- Flask Application Initialization ---
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))

# --- Load Environment Variables ---
# This will load variables from a .env file (e.g., API keys)
load_dotenv()

# --- Set API Keys as Environment Variables ---
# Pinecone API Key and Environment (Region)
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_ENVIRONMENT"] = os.getenv("PINECONE_API_ENV", "us-east-1") # Default to us-east-1 if not explicitly set

# OpenAI API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# --- Initialize Models and RAG Chain ---
# 1. Download/Load Embedding Model
# This model converts text into vector embeddings
embedding_model = download_hugging_face_embedding_model()

# 2. Load Existing Pinecone Index as Retriever
# The index name used in the demo is 'medical-bot'
index_name = "medical-bot"
pinecone_vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)

# Create a retriever from the Pinecone vector store
# The demo showed retrieving 3 relevant documents
retriever = pinecone_vector_store.as_retriever(search_kwargs={"k": 3})

# 3. Initialize Large Language Model (LLM)
# Using OpenAI's 'gpt-3.5-turbo' model with temperature 0 for deterministic output
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4,
    max_tokens=500
)

# 4. Create the Chat Prompt Template
# Combines the system prompt (from src/prompt.py) with the user's input
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt), # The system prompt guiding the chatbot
    ("human", "{input}") # Placeholder for user's query
])

# 5. Create the Document Chain
# This chain takes the retrieved documents and the user's question, then formats it for the LLM
document_chain = create_stuff_documents_chain(llm, prompt_template)

# 6. Create the Retrieval Chain (RAG Chain)
# This chain combines the retriever and the document chain to perform RAG
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- Flask Routes ---

# Default route: Renders the chat user interface
@app.route('/')
def index():
    """
    Renders the main chat interface (index.html).
    """
    return render_template('index.html')

# Chat route: Handles user queries and returns chatbot responses
@app.route('/chat', methods=['POST'])
def chat():
    """
    Receives user messages, processes them using the RAG chain,
    and returns the chatbot's response in the appropriate language.
    """
    user_message = request.json.get('message')

    if not user_message:
        return jsonify({'response': 'Please enter a message.'}), 400

    # Detect the language of the user's query
    detected_language = detect_language(user_message)
    
    try:
        # Preprocess the text if it's Hindi
        if detected_language == 'hi':
            user_message = preprocess_hindi_text(user_message)
        
        # Select appropriate system prompt based on detected language
        if detected_language == 'hi':
            current_system_prompt = system_prompt_hi
        else:
            current_system_prompt = system_prompt_en
        
        # Create a new prompt template with the appropriate system prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", current_system_prompt),
            ("human", "{input}")
        ])
        
        # Create a new document chain with the language-specific prompt
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        
        # Create a new retrieval chain with the language-specific document chain
        retrieval_chain_lang = create_retrieval_chain(retriever, document_chain)
        
        # Invoke the RAG chain with the user's query
        response = retrieval_chain_lang.invoke({"input": user_message})
        bot_response = response["answer"]
        
        return jsonify({'response': bot_response, 'language': detected_language})
    except Exception as e:
        print(f"Error processing chat message: {e}")
        error_msg = "Sorry, an error occurred. Please try again."
        if detected_language == 'hi':
            error_msg = "क्षमा करें, एक त्रुटि हुई। कृपया पुनः प्रयास करें।"
        return jsonify({'response': error_msg}), 500
    

print("Templates path:", os.path.abspath("templates"))

# --- Run the Flask Application ---
if __name__ == '__main__':
    """
    Runs the Flask application on Local Host at port 8080.
    Debug mode is enabled for automatic updates on code changes.
    """
    app.run(host="0.0.0.0", port=8080, debug=True)