from flask import Flask, render_template, request, jsonify # Flask web framework components for web application
from SRC.helper import download_hugging_face_embeddings # Import function to download embedding model
from langchain_pinecone import PineconeVectorStore # For interacting with Pinecone vector database
from langchain_openai import OpenAI # For using OpenAI's Large Language Model
from langchain.prompts import PromptTemplate # For creating prompt templates
from langchain.chains.combine_documents import create_stuff_documents_chain # For combining retrieved documents with a prompt
from langchain.chains import create_retrieval_chain # For creating a retrieval-augmented generation (RAG) chain
from dotenv import load_dotenv # For loading environment variables
import os # For interacting with the operating system (e.g., setting environment variables)
from SRC.prompt import system_prompt_template # Import the system prompt template from src/prompt.py
from langchain_groq import ChatGroq

# Initialize the Flask application
app = Flask(__name__,template_folder="templates")

# Load environment variables from the .env file
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GROQ_API_KEY"]=GROQ_API_KEY

embeddings = download_hugging_face_embeddings()

index_name="medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 5})

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4,
    max_tokens=500)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt=(
    "you are an assistant for question-answer."
    "use the following peices of retrieved context to answer "
    "the question.if you dont know the answer,say that you "
    "dont know .use three sentences maximum and keep the "
    "answer concise. "
    "\n\n"
    "{context}"
)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}"),
    ]
)
question_answer_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    input=msg
    print(input)
    response=rag_chain.invoke({"input":msg})
    print("Response:",response["answer"])
    return str(response["answer"])


if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)