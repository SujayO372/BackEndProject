import glob
import os
import logging
import json
from datetime import datetime
from pinecone import Pinecone  
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

import google.generativeai as genai
import pdfplumber 
from markdownify import markdownify as md
from pinecone_plugins.assistant.models.chat import Message

# Load .env variables
load_dotenv(dotenv_path='.env')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check API keys
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set")
if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY must be set")

if not os.environ.get("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY is not set")

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


def call_gemini_api(prompt_text: str) -> str:
    try:
        # Create a Gemini model instance, e.g. gemini-1.5-flash
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Generate content by passing the prompt
        response = model.generate_content(prompt_text)
        
        # Return the generated text
        return response.text.strip()

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return ""
    

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


    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=os.get(PINECONE_API_KEY))
    assistant = pc.assistant.Assistant(assistant_name="pineconeai")
    msg = Message(role="user", content=sanitized_query)
    resp = assistant.chat(messages=[msg])
    response = jsonify({
        "response": {
            "query": sanitized_query,
            "result": resp
        }
    })
    return response

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

@application.route('/pinecone', methods=['OPTIONS', 'POST'])
def pinecone():
    dummy_message = "hi"
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=os.get(PINECONE_API_KEY))
    assistant = pc.assistant.Assistant(assistant_name="example-assistant")
    msg = Message(role="user", content=dummy_message)
    resp = assistant.chat(messages=[msg])
    response = jsonify({
        "response": {
            "query": dummy_message,
            "result": resp
        }
    })
    return response
# Test Supabase
@application.route('/test-db', methods=['GET'])
def test_database():
    try:
        result = supabase.table('_realtime_schema').select('*').limit(1).execute()
        return jsonify({"status": "success", "message": "Database connected successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Database connection failed: {str(e)}"}), 500

# Updated /health-test endpoint
@application.route('/health-test', methods=['OPTIONS', 'POST'])
def health_test():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = frontend_url
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response, 204

    print("--------- inside healthtest...")
    load_dotenv(dotenv_path='.env')  # Optional: specify path explicitly
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    if api_key:
        print("GEMINI_API_KEY loaded.", api_key)
    else:
        print("GEMINI_API_KEY missing.")

    import google.generativeai as genai

    # Configure the gemini client with your key
    genai.configure(api_key=api_key)
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    answers = data.get("answers")
    if not answers or not isinstance(answers, dict):
        return jsonify({"error": "Missing or invalid 'answers' field"}), 400

    # Combine all text for crisis check
    combined_text = " ".join(str(v) for v in answers.values())
    if detect_crisis(combined_text):
        return jsonify({
            "response": {
                "result": CRISIS_RESPONSE,
                "recommendations": []
            }
        })

    # Build Gemini prompt
    prompt = f"""
You are a helpful mental health assistant.
Based on these user answers to a health checkup, suggest 3 relevant mental health articles.

Return the result in this JSON format:
[
  {{
    "title": "Article Title",
    "summary": "Brief summary",
    "link": "https://example.com/article"
  }}
]

User Answers:
{json.dumps(answers, indent=2)}
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        text_response = response.text.strip()

        # Attempt to parse JSON recommendations from Gemini response
        try:
            recommendations = json.loads(text_response)
        except json.JSONDecodeError:
            logging.warning("Failed to parse Gemini JSON response. Returning raw text as summary.")
            recommendations = [{
                "title": "Unable to parse recommendations",
                "summary": text_response,
                "link": "#"
            }]

        return jsonify({
            "response": {
                "result": "Here are your recommended resources.",
                "recommendations": recommendations
            }
        })

    except Exception as e:
        logging.exception("Gemini health test error:")
        return jsonify({"error": str(e)}), 500

# pinecone 

def process_file(file):
    text = file.read().decode("utf-8")
    chunks = split_text(text)

    vectors = []
    for i, chunk in enumerate(chunks):
        vector_id = f"{file.name}_{i}_{uuid4().hex[:8]}"
        embedding = get_embedding(chunk)
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {"text": chunk, "source": file.name}
        })

    index.upsert(vectors=vectors)
    return len(vectors)


def convert_pdf_to_markdown(pdf_path, output_dir):
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    md_path = os.path.join(output_dir, f"{filename}.md")

    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"

    markdown_text = md(full_text)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text) 
    process_file(f)
    print(f"Converted: {pdf_path} → {md_path}")

def batch_convert_pdfs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, file)
            convert_pdf_to_markdown(pdf_path, output_dir)

# === Set your paths here ===

batch_convert_pdfs()


input_directory = "/path/to/pdf/folder"
output_directory = "/path/to/markdown/output"

batch_convert_pdfs(input_directory, output_directory)

# To use the Python SDK, install the plugin:
# pip install --upgrade pinecone pinecone-plugin-assistant





if __name__ == '__main__':
    application.run(debug=True)

