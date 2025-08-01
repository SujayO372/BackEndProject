import glob, os
import logging
import json  # Added missing import
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
CORS(application, origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"])  # Vite runs on 5173

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

# Initialize Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    logging.warning("SUPABASE_URL and SUPABASE_KEY not set - database features will be disabled")
    supabase = None
else:
    supabase: Client = create_client(supabase_url, supabase_key)

# Initialize embedding model and vector store
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)
vector_store = InMemoryVectorStore(embeddings)

# Load documents
def load_documents():
    try:
        text_files = glob.glob("documents/*.txt")
        if not text_files:
            logging.warning("No text files found in documents/ directory")
            return
        
        docs = []
        for text_path in text_files:
            try:
                loader = TextLoader(text_path)
                chunk = loader.load()
                docs.extend(chunk)
            except Exception as e:
                logging.warning(f"Could not load {text_path}: {e}")
        
        if docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            all_splits = text_splitter.split_documents(docs)
            vector_store.add_documents(all_splits)
            logging.info(f"Loaded {len(all_splits)} document chunks")
        else:
            logging.warning("No documents were loaded")
    except Exception as e:
        logging.error(f"Error loading documents: {e}")

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

CRISIS_RESPONSE = """Crisis Support Detected - Please reach out for help immediately:
- Call 988 (Suicide & Crisis Lifeline)
- Text "HELLO" to 741741 (Crisis Text Line)
- Call 911 for emergencies
- Visit your nearest emergency room

You are not alone. Professional help is available 24/7."""

def add_disclaimer(response):
    disclaimer = "\n\n**Disclaimer:** I am not a licensed therapist. Please consult a professional for serious concerns."
    return response + disclaimer

# Endpoint: /query (for OpenAI/LangChain)
@application.route('/query', methods=['OPTIONS', 'POST'])
def query():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response, 204

    try:
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

        return jsonify({
            "response": {
                "query": user_query,
                "result": answer
            }
        })
    except Exception as e:
        logging.exception("Error in /query endpoint:")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Endpoint: /gemini-query (for Gemini AI)
@application.route('/gemini-query', methods=['OPTIONS', 'POST'])
def gemini_query():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response, 204

    try:
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

        # Enhanced prompt for mental health context
        mental_health_prompt = f"""
        You are a compassionate mental health support assistant. Please provide helpful, supportive, and informative responses about mental health topics.

        User Question: {sanitized_query}

        Please provide a thoughtful response that:
        1. Shows empathy and understanding
        2. Provides practical advice when appropriate
        3. Encourages professional help when needed
        4. Is supportive but not diagnostic

        Response:
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(mental_health_prompt)

        answer_text = response.text.strip()
        answer_text = add_disclaimer(answer_text)

        return jsonify({
            "response": {
                "query": user_query,
                "result": answer_text,
                "sources": []
            }
        })

    except Exception as e:
        logging.exception("Gemini API error:")
        return jsonify({"error": f"AI service temporarily unavailable: {str(e)}"}), 500

# Endpoint: /health-test (for mental health assessment)
@application.route('/health-test', methods=['OPTIONS', 'POST'])
def health_test():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response, 204

    try:
        data = request.get_json()
        answers = data.get("answers")

        if not answers:
            return jsonify({"error": "Missing health test answers"}), 400

        # Convert answers to string values for analysis
        answer_texts = []
        for key, value in answers.items():
            answer_texts.append(f"Question {key}: {value}")
        
        combined_text = " ".join(answer_texts)
        
        # Check for crisis indicators
        if detect_crisis(combined_text):
            return jsonify({
                "response": {
                    "result": CRISIS_RESPONSE,
                    "recommendations": []
                }
            })

        # Create a comprehensive assessment prompt
        assessment_prompt = f"""
        You are a mental health assessment assistant. Based on the following mental health questionnaire responses, provide personalized recommendations and insights.

        Assessment Responses:
        {json.dumps(answers, indent=2)}

        Please analyze these responses and provide:
        1. A brief, supportive summary of the person's current state
        2. 3-5 specific, actionable recommendations tailored to their responses
        3. Any important considerations or suggestions for professional support

        Format your response as practical, encouraging advice. Focus on specific actions they can take to improve their mental well-being.

        Response:
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content(assessment_prompt)
        
        assessment_result = result.text.strip()
        assessment_result = add_disclaimer(assessment_result)

        # Try to extract structured recommendations if possible
        recommendations = []
        try:
            # Simple parsing to extract bullet points or numbered items as recommendations
            lines = assessment_result.split('\n')
            for line in lines:
                line = line.strip()
                if (line.startswith('•') or line.startswith('-') or 
                    line.startswith('*') or any(line.startswith(f'{i}.') for i in range(1, 10))):
                    clean_rec = line.lstrip('•-*0123456789. ').strip()
                    if clean_rec and len(clean_rec) > 10:  # Only meaningful recommendations
                        recommendations.append({
                            "title": "Personalized Recommendation",
                            "summary": clean_rec,
                            "link": "#"
                        })
        except:
            pass

        return jsonify({
            "response": {
                "result": assessment_result,
                "recommendations": recommendations[:5]  # Limit to 5 recommendations
            }
        })

    except Exception as e:
        logging.exception("Health test error:")
        return jsonify({"error": f"Assessment service temporarily unavailable: {str(e)}"}), 500

# Health check endpoint
@application.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "services": {
            "gemini": "available" if os.environ.get("GEMINI_API_KEY") else "disabled",
            "openai": "available" if os.environ.get("OPENAI_API_KEY") else "disabled",
            "supabase": "available" if supabase else "disabled"
        }
    })

# Test database connection
@application.route('/test-db', methods=['GET'])
def test_database():
    if not supabase:
        return jsonify({"status": "error", "message": "Supabase not configured"}), 500
    
    try:
        result = supabase.table('_realtime_schema').select('*').limit(1).execute()
        return jsonify({"status": "success", "message": "Database connected successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Database connection failed: {str(e)}"}), 500

# Root endpoint
@application.route('/', methods=['GET'])
def root():
    return jsonify({
        "message": "Mental Health Platform API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/gemini-query",
            "/health-test",
            "/query",
            "/test-db"
        ],
        "status": "running"
    })

if __name__ == '__main__':
    print("🚀 Starting Mental Health Platform Backend...")
    print("📍 Backend will be available at:")
    print("   - http://localhost:5000")
    print("   - http://127.0.0.1:5000")
    print("")
    print("🌐 Frontend (Vite) should be running on: http://localhost:5173")
    print("🔗 Frontend should connect to: http://localhost:5000")
    print("")
    print("📋 Available endpoints:")
    print("   - GET  /health")
    print("   - POST /gemini-query")
    print("   - POST /health-test")
    print("   - POST /query")
    print("   - GET  /test-db")
    print("")
    
    # Bind to localhost specifically to match frontend expectations
    application.run(debug=True, port=5000, host='localhost')