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
import re

# use TextLoader instead of PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from supabase import create_client, Client

import google.generativeai as genai
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
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt_text)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
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
Try to keep it within 1-2 paragraphs, and be concise and supportive.
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

frontend_url = 'http://www.serenespaceai.com'

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


def cors_response(json_data, status=200):
    response = jsonify(json_data)
    response.headers['Access-Control-Allow-Origin'] = frontend_url
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    return response, status


# Endpoint: /query
@application.route('/query', methods=['OPTIONS', 'POST'])
def query():
    if request.method == 'OPTIONS':
        return cors_response({}, 204)

    data = request.json
    user_query = data.get("query")
    if not user_query:
        return cors_response({"error": "Query parameter is required"}, 400)

    sanitized_query, error = validate_and_sanitize_input(user_query)
    if error:
        return cors_response({"error": error}, 400)

    if detect_crisis(sanitized_query):
        logging.warning(f"Crisis detected: {sanitized_query[:50]}...")
        return cors_response({"response": {"query": user_query, "result": CRISIS_RESPONSE}})

    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    assistant = pc.assistant.Assistant(assistant_name="pineconeai")
    msg = Message(role="user", content=sanitized_query)
    resp = assistant.chat(messages=[msg])
    return cors_response({
        "response": {
            "query": sanitized_query,
            "result": resp
        }
    })


# Health check
@application.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


# Endpoint: /pinecone
@application.route('/pinecone', methods=['OPTIONS', 'POST'])
def pinecone():
    if request.method == 'OPTIONS':
        return cors_response({}, 204)

    dummy_message = "hi"
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    assistant = pc.assistant.Assistant(assistant_name="example-assistant")
    msg = Message(role="user", content=dummy_message)
    resp = assistant.chat(messages=[msg])
    return cors_response({
        "response": {
            "query": dummy_message,
            "result": resp
        }
    })


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
        return cors_response({}, 204)

    data = request.get_json()
    print(f"health-test called with data: {data}")

    if not data or not isinstance(data.get("answers"), dict):
        return cors_response({"error": "Missing or invalid 'answers' field"}, 400)

    answers = data["answers"]
    print(f"User answers: {answers}")
    combined_text = " ".join(str(v) for v in answers.values())
    if detect_crisis(combined_text):
        return cors_response({
            "response": {
                "result": CRISIS_RESPONSE,
                "recommendations": []
            }
        })

    prompt = (
        "You are a helpful mental health assistant.\n"
        "Based on these user answers to a health checkup, suggest 3 relevant mental health articles.\n"
        "Return the result in this JSON format:\n"
        "[\n"
        "  {\n"
        "    \"title\": \"Article Title\",\n"
        "    \"summary\": \"Brief summary\",\n"
        "    \"link\": \"https://example.com/article\"\n"
        "  }\n"
        "]\n\n"
        " Try to get a link from an actual article from the internet.\n"
        f"User Answers:\n{json.dumps(answers, indent=2)}"
    )

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        text_response = response.text.strip()

        # Try to extract JSON array from response
        match = re.search(r"\[(\s*{.*?}\s*)\]", text_response, re.DOTALL)
        if match:
            json_content = match.group(0)
            try:
                recommendations = json.loads(json_content)
            except Exception:
                recommendations = []
        else:
            recommendations = []

        if not recommendations:
            recommendations = [{
                "title": "General Mental Health Support",
                "summary": "Basic tips to improve mental wellbeing.",
                "link": "https://www.mentalhealth.gov/"
            }]

        return cors_response({
            "response": {
                "result": "Recommendations generated",
                "recommendations": recommendations
            }
        })
    except Exception as e:
        logging.error(f"Error generating health test recommendations: {e}")
        return cors_response({
            "response": {
                "result": "Error generating recommendations.",
                "recommendations": []
            }
        }, 500)


# Endpoint: /get-topic
@application.route("/get-topic", methods=["OPTIONS", "POST"])
def get_topic():
    if request.method == "OPTIONS":
        return cors_response({}, 204)

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    logging.info(f"get-topic called with text: {text}")

    if not text:
        return cors_response({"topic": "Untitled Chat"})

    def clean_and_correct_text(input_text):
        corrections = {
            'anxeity': 'anxiety', 'aniety': 'anxiety', 'anexiety': 'anxiety',
            'depresion': 'depression', 'depress': 'depression', 'depresed': 'depressed',
            'stres': 'stress', 'stresed': 'stressed', 'overwelmed': 'overwhelmed',
            'panick': 'panic', 'panik': 'panic', 'anxius': 'anxious',
            'lonley': 'lonely', 'lonly': 'lonely', 'isloated': 'isolated',
            'slepp': 'sleep', 'slep': 'sleep', 'insomia': 'insomnia',
            'relatinship': 'relationship', 'relashionship': 'relationship',
            'therapist': 'therapist', 'counceling': 'counseling', 'councilor': 'counselor',
            'mindfulnes': 'mindfulness', 'mediation': 'meditation',
            'addication': 'addiction', 'adiction': 'addiction',
            'tramatic': 'traumatic', 'truama': 'trauma', 'trama': 'trauma'
        }

        words = input_text.lower().split()
        corrected_words = []
        for word in words:
            clean_word = word.strip('.,!?;:"()[]{}')
            corrected_words.append(corrections.get(clean_word, word))
        return " ".join(corrected_words)

    def capitalize_every_word(title_str):
        return " ".join(word.capitalize() for word in title_str.split())

    try:
        cleaned_text = clean_and_correct_text(text)

        prompt = f"""You are an expert mental health chatbot. Generate a precise 2-3 word title that summarizes the core emotional need or topic from this message.

Key Requirements:
- Use warm, supportive language (not clinical)
- Focus on help/support rather than problems
- Be concise but meaningful
- Avoid medical terminology

Input: "{cleaned_text[:300]}"

Quality Examples:
"I keep having panic attacks at work" → "Workplace Anxiety"
"My depression is getting worse lately" → "Depression Care"
"How can I improve my self confidence" → "Building Confidence"
"I can't sleep because my mind races" → "Sleep Support"
"My relationship with my partner is falling apart" → "Relationship Guidance"
"I feel completely overwhelmed with life" → "Life Balance"
"I'm grieving the loss of my mother" → "Grief Healing"
"I think I might have PTSD from trauma" → "Trauma Recovery"
"I want to learn meditation techniques" → "Mindfulness Training"
"I'm struggling with alcohol addiction" → "Addiction Recovery"

Generate title:"""

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        title = response.text.strip()

        # Clean title of unwanted punctuation and prefixes
        title = title.strip('"\'.,!?:').strip()
        prefixes = ['title:', 'topic:', 'generate title:', 'response:', 'answer:']
        for prefix in prefixes:
            if title.lower().startswith(prefix):
                title = title[len(prefix):].strip()
                break

        # Validate and format title
        if title and 3 <= len(title) <= 45 and not title.lower().startswith('i '):
            title = capitalize_every_word(title)
        else:
            text_lower = cleaned_text.lower()
            if any(term in text_lower for term in ['anxiety', 'anxious', 'panic', 'worried', 'nervous']):
                title = "Anxiety Support"
            elif any(term in text_lower for term in ['depression', 'depressed', 'sad', 'hopeless', 'down']):
                title = "Mood Support"
            elif any(term in text_lower for term in ['stress', 'overwhelmed', 'burnout']):
                title = "Stress Management"
            elif any(term in text_lower for term in ['trauma', 'ptsd', 'abuse']):
                title = "Trauma Recovery"
            else:
                title = "Mental Health Support"

        return cors_response({"topic": title})

    except Exception as e:
        logging.error(f"Error in /get-topic: {e}")
        return cors_response({"topic": "Untitled Chat"})





if __name__ == '__main__':
    application.run(host="0.0.0.0", port=5000, debug=True)
