import glob, os
import logging
from datetime import datetime

from flask import Flask, request, jsonify, make_response
from dotenv import load_dotenv
from flask_cors import CORS
from typing_extensions import List, TypedDict

# use TextLoader instead of PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from supabase import create_client, Client
import requests
from pinecone import Pinecone, ServerlessSpec
import pinecone
import google.generativeai as genai

# Load .env variables
load_dotenv(dotenv_path='.env')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check API keys
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set")
if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY must be set")

# Configure Gemini client
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

application = Flask(__name__)
CORS(application)  # CORS enabled for frontend

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

# Initialize Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize embedding model and vector store
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)
vector_store = InMemoryVectorStore(embeddings)

# Load documents

def load_documents():
    text_files = glob.glob("documents/*.txt")
    docs = []
    for text_path in text_files:
        loader = TextLoader(text_path)
        chunk = loader.load()
        docs.extend(chunk)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(all_splits)

load_documents()

# Prompt template
prompt_template = """
You are a helpful mental health assistant chatbot. Use the following context to answer the user's question.
Context:
{context}

Question: {question}

Answer:
""".strip()

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    final_prompt = prompt_template.format(
        context=docs_content,
        question=state["question"]
    )
    response = llm.invoke([{"role": "user", "content": final_prompt}])
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

frontend_url = 'http://localhost:5173'

# Safety functions
def validate_and_sanitize_input(query):
    if len(query) > 1000:
        return None, "Message too long."
    return query.strip(), None

CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end it all', 'harm myself', 'hurt myself',
    'want to die', 'worthless', 'hopeless', 'cutting', 'overdose'
]

def detect_crisis(query):
    return any(keyword in query.lower() for keyword in CRISIS_KEYWORDS)

CRISIS_RESPONSE = """ As an AI Chatbot, I hold concern for you. Please reach out for help:
- Take a look at the Hotlines page, and dial any numbers.
- Call 988 (Suicide & Crisis Lifeline)
- Text "HELLO" to 741741 (Crisis Text Line)
- Call 911 for emergencies"""

NOTENOUGH_INFORMATION = """
As a Mental Health subjected Chatbot, I am unable to answer this question, as I do not have any context related to it.
"""

def add_disclaimer(response):
    disclaimer = "\n\n**Disclaimer:** I am not a licensed therapist. Please consult a professional for serious concerns."
    return response + disclaimer

# Endpoint: /query
@application.route('/query', methods=['OPTIONS', 'POST'])
def query():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = frontend_url
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response, 204

    data = request.json
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400

    sanitized_query, error = validate_and_sanitize_input(user_query)
    if error:
        return jsonify({"error": error}), 400

    if detect_crisis(sanitized_query):
        logging.warning(f"Crisis detected: {sanitized_query[:50]}...")
        return jsonify({"response": {"query": user_query, "result": CRISIS_RESPONSE}})

    final_state = graph.invoke({"question": sanitized_query})
    answer = add_disclaimer(final_state["answer"].strip())

    response = jsonify({
        "response": {
            "query": user_query,
            "result": answer
        }
    })
    response.headers['Access-Control-Allow-Origin'] = frontend_url
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    return response

# Endpoint: /gemini-query
@application.route('/gemini-query', methods=['OPTIONS', 'POST'])
def gemini_query():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = frontend_url
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response, 204

    data = request.json
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400

    sanitized_query, error = validate_and_sanitize_input(user_query)
    if error:
        return jsonify({"error": error}), 400

    if detect_crisis(sanitized_query):
        logging.warning(f"Crisis detected in Gemini: {sanitized_query[:50]}...")
        return jsonify({
            "response": {
                "query": user_query,
                "result": CRISIS_RESPONSE,
                "sources": []
            }
        })

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(sanitized_query)

        answer_text = response.text.strip()
        answer_text = add_disclaimer(answer_text)

        response_data = {
            "response": {
                "query": user_query,
                "result": answer_text,
                "sources": []
            }
        }

        response = jsonify(response_data)
        response.headers['Access-Control-Allow-Origin'] = frontend_url
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response

    except Exception as e:
        logging.exception("Gemini API error:")
        return jsonify({"error": str(e)}), 500

# Health check
@application.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

# Test Supabase
@application.route('/test-db', methods=['GET'])
def test_database():
    try:
        result = supabase.table('_realtime_schema').select('*').limit(1).execute()
        return jsonify({"status": "success", "message": "Database connected successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Database connection failed: {str(e)}"}), 500

if __name__ == '__main__':
    application.run(debug=True)
