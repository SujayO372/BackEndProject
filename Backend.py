import glob
import os
import logging
import json
import re
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

from pinecone_plugins.assistant.models.chat import Message

# Load .env variables
load_dotenv(dotenv_path='.env')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
print("OPENAI_API_KEY:", "OPENAI_API_KEY" in os.environ)
# Check API keys with better error messages
def check_required_env_vars():
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key is missing",
        "PINECONE_API_KEY": "Pinecone API key is missing",
        "SUPABASE_URL": "Supabase URL is missing",
        "SUPABASE_KEY": "Supabase key is missing"
    }
    
    missing_vars = []
    for var, message in required_vars.items():
        if not os.environ.get(var):
            missing_vars.append(f"{var}: {message}")
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables:\n" + "\n".join(missing_vars))

def validate_api_keys():
    """Validate API keys by testing them"""
    # Test OpenAI API key
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and len(openai_key) > 10:
        logging.info("OpenAI API key format appears valid")
    else:
        logging.warning("OpenAI API key may be invalid")

check_required_env_vars()

application = Flask(__name__)
CORS(application)  # CORS enabled for frontend

# Validate API keys on startup
validate_api_keys()

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

# Initialize Supabase with better error handling
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

try:
    supabase: Client = create_client(supabase_url, supabase_key)
    logging.info("Supabase client initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Supabase client: {e}")
    supabase = None

# Initialize embedding model and vector store
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)
vector_store = InMemoryVectorStore(embeddings)

# Load documents with better error handling
def load_documents():
    """Load and process text documents"""
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
                logging.info(f"Loaded document: {text_path}")
            except Exception as e:
                logging.error(f"Failed to load document {text_path}: {e}")
        
        if docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            all_splits = text_splitter.split_documents(docs)
            vector_store.add_documents(all_splits)
            logging.info(f"Successfully loaded {len(all_splits)} document chunks")
        else:
            logging.warning("No documents were successfully loaded")
    except Exception as e:
        logging.error(f"Error in load_documents: {e}")

# Load documents on startup
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
    """Retrieve relevant documents"""
    try:
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}
    except Exception as e:
        logging.error(f"Error in retrieve: {e}")
        return {"context": []}

def generate(state: State):
    """Generate response using LLM"""
    try:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        final_prompt = prompt_template.format(
            context=docs_content,
            question=state["question"]
        )
        response = llm.invoke([{"role": "user", "content": final_prompt}])
        return {"answer": response.content}
    except Exception as e:
        logging.error(f"Error in generate: {e}")
        return {"answer": "I'm sorry, I'm having trouble generating a response right now."}

# Build the graph
try:
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    logging.info("Graph compiled successfully")
except Exception as e:
    logging.error(f"Error building graph: {e}")
    graph = None

frontend_url = 'http://localhost:5173'

# Safety functions
def validate_and_sanitize_input(query):
    """Validate and sanitize user input"""
    if not query or not isinstance(query, str):
        return None, "Invalid query format"
    
    query = query.strip()
    if len(query) > 1000:
        return None, "Message too long (max 1000 characters)"
    
    if len(query) < 1:
        return None, "Message too short"
    
    return query, None

CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end it all', 'harm myself', 'hurt myself',
    'want to die', 'worthless', 'hopeless', 'cutting', 'overdose',
    'self harm', 'suicidal', 'kill me', 'death wish', 'no point living'
]

def detect_crisis(query):
    """Detect crisis-related content in user query"""
    if not query:
        return False
    return any(keyword in query.lower() for keyword in CRISIS_KEYWORDS)

CRISIS_RESPONSE = """I'm concerned about what you've shared. Please reach out for immediate help:

🆘 **Emergency Resources:**
- Call 988 (Suicide & Crisis Lifeline) - Available 24/7
- Text "HELLO" to 741741 (Crisis Text Line)
- Call 911 for immediate emergencies
- Visit your nearest emergency room

🤝 **You are not alone.** Professional counselors are available to help you through this difficult time.

**Disclaimer:** I am not a licensed therapist. Please consult a professional for serious concerns."""

def add_disclaimer(response):
    """Add disclaimer to responses"""
    disclaimer = "\n\n**Disclaimer:** I am not a licensed therapist. Please consult a professional for serious concerns."
    return response + disclaimer

def cors_response(json_data, status=200):
    """Create CORS-enabled response"""
    response = jsonify(json_data)
    response.headers['Access-Control-Allow-Origin'] = frontend_url
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS, GET'
    return response, status

# Health check endpoint
@application.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "services": {
            "supabase": supabase is not None,
            "vector_store": vector_store is not None,
            "graph": graph is not None
        }
    })

# Main query endpoint
@application.route('/query', methods=['OPTIONS', 'POST'])
def query():
    """Main chatbot query endpoint"""
    if request.method == 'OPTIONS':
        return cors_response({}, 204)

    try:
        data = request.get_json()
        if not data:
            return cors_response({"error": "No JSON data provided"}, 400)
        
        user_query = data.get("query")
        if not user_query:
            return cors_response({"error": "Query parameter is required"}, 400)

        # Validate and sanitize input
        sanitized_query, error = validate_and_sanitize_input(user_query)
        if error:
            return cors_response({"error": error}, 400)

        # Check for crisis content
        if detect_crisis(sanitized_query):
            logging.warning(f"Crisis detected: {sanitized_query[:50]}...")
            return cors_response({
                "response": {
                    "query": user_query, 
                    "result": CRISIS_RESPONSE
                }
            })

        # Try Pinecone assistant first
        try:
            PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            assistant = pc.assistant.Assistant(assistant_name="pineconeai")
            msg = Message(role="user", content=sanitized_query)
            resp = assistant.chat(messages=[msg])
            
            # Add disclaimer to response
            if hasattr(resp, 'message') and hasattr(resp.message, 'content'):
                result = add_disclaimer(resp.message.content)
            else:
                result = add_disclaimer(str(resp))
            
            return cors_response({
                "response": {
                    "query": sanitized_query,
                    "result": result
                }
            })
        except Exception as pinecone_error:
            logging.error(f"Pinecone assistant error: {pinecone_error}")
            
            # Fallback to local RAG if available
            if graph:
                try:
                    state = {"question": sanitized_query}
                    result = graph.invoke(state)
                    response_text = add_disclaimer(result.get("answer", "I'm sorry, I couldn't generate a response."))
                    
                    return cors_response({
                        "response": {
                            "query": sanitized_query,
                            "result": response_text
                        }
                    })
                except Exception as graph_error:
                    logging.error(f"Graph processing error: {graph_error}")
            
            # Final fallback
            return cors_response({
                "response": {
                    "query": sanitized_query,
                    "result": add_disclaimer("I'm experiencing technical difficulties. Please try again later or contact support if the issue persists.")
                }
            }, 503)

    except Exception as e:
        logging.error(f"Unexpected error in /query: {e}")
        return cors_response({"error": "Internal server error"}, 500)

# Pinecone test endpoint
@application.route('/pinecone', methods=['OPTIONS', 'POST'])
def pinecone_test():
    """Test Pinecone connection"""
    if request.method == 'OPTIONS':
        return cors_response({}, 204)

    try:
        dummy_message = "hi"
        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        assistant = pc.assistant.Assistant(assistant_name="example-assistant")
        msg = Message(role="user", content=dummy_message)
        resp = assistant.chat(messages=[msg])
        
        return cors_response({
            "response": {
                "query": dummy_message,
                "result": str(resp)
            }
        })
    except Exception as e:
        logging.error(f"Pinecone test error: {e}")
        return cors_response({"error": f"Pinecone test failed: {str(e)}"}, 500)

# Test database connection
@application.route('/test-db', methods=['GET'])
def test_database():
    """Test database connection"""
    if not supabase:
        return jsonify({"status": "error", "message": "Supabase client not initialized"}), 500
    
    try:
        # Try a simple query that should work on any Supabase instance
        result = supabase.table('_realtime_schema').select('*').limit(1).execute()
        return jsonify({"status": "success", "message": "Database connected successfully!"})
    except Exception as e:
        logging.error(f"Database test error: {e}")
        return jsonify({"status": "error", "message": f"Database connection failed: {str(e)}"}), 500

# Health test endpoint using OpenAI
@application.route('/health-test', methods=['OPTIONS', 'POST'])
def health_test():
    """Generate health recommendations based on user answers"""
    if request.method == 'OPTIONS':
        return cors_response({}, 204)

    try:
        data = request.get_json()
        # if not data or not isinstance(data.get("answers"), dict):
        #     return cors_response({"error": "Missing or invalid 'answers' field"}, 400)

        answers = data["answers"]
        combined_text = " ".join(str(v) for v in answers.values())
        
        # Check for crisis content
        if detect_crisis(combined_text):
            return cors_response({
                "response": {
                    "result": CRISIS_RESPONSE,
                    "recommendations": []
                }
            })

        # Use OpenAI to generate recommendations
        try:
            recommendations = generate_openai_recommendations(answers)
            if not recommendations:
                raise Exception("OpenAI returned no recommendations")
            
            return cors_response({
                "response": {
                    "result": "Recommendations generated successfully",
                    "recommendations": recommendations
                }
            })
        except Exception as e:
            logging.error(f"Error generating recommendations with OpenAI: {e}")
            # Fall back to smart keyword-based recommendations
            recommendations = get_smart_fallback_recommendations(answers)
            return cors_response({
                "response": {
                    "result": "Recommendations generated successfully (fallback)",
                    "recommendations": recommendations
                }
            })
        
    except Exception as e:
        logging.error(f"Unexpected error in /health-test: {e}")
        return cors_response({
            "response": {
                "result": "Error generating recommendations.",
                "recommendations": get_fallback_recommendations()
            }
        }, 200)

def generate_openai_recommendations(answers):
    """Generate recommendations using OpenAI"""
    try:
        prompt = f"""You are a helpful mental health assistant. Based on the user's answers to a mental health checkup, suggest 3 relevant mental health resources.

Return your response in this exact JSON format:
[
  {{
    "title": "Resource Title",
    "summary": "Brief helpful summary",
    "link": "https://example.com/resource"
  }}
]

Guidelines:
- Suggest real, reputable mental health resources (NIMH, APA, mental health organizations)
- Focus on evidence-based information
- Make summaries supportive and encouraging
- Use actual working links when possible

User's answers: {json.dumps(answers, indent=2)}

Response (JSON only):"""

        response = llm.invoke([{"role": "user", "content": prompt}])
        response_text = response.content.strip()
        
        # Try to extract JSON from the response
        match = re.search(r'\[[\s\S]*\]', response_text)
        if match:
            json_content = match.group(0)
            try:
                recommendations = json.loads(json_content)
                # Validate structure
                if isinstance(recommendations, list) and len(recommendations) > 0:
                    for rec in recommendations:
                        if not all(key in rec for key in ['title', 'summary', 'link']):
                            raise ValueError("Missing required fields")
                    return recommendations[:3]  # Limit to 3
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Failed to parse OpenAI JSON: {e}")
        
        # If JSON parsing failed, try to extract information manually
        lines = response_text.split('\n')
        recommendations = []
        current_rec = {}
        
        for line in lines:
            line = line.strip()
            if '"title":' in line:
                title = line.split('"title":')[1].strip().strip('",')
                current_rec['title'] = title
            elif '"summary":' in line:
                summary = line.split('"summary":')[1].strip().strip('",')
                current_rec['summary'] = summary
            elif '"link":' in line:
                link = line.split('"link":')[1].strip().strip('",')
                current_rec['link'] = link
                if all(key in current_rec for key in ['title', 'summary', 'link']):
                    recommendations.append(current_rec.copy())
                    current_rec = {}
        
        if len(recommendations) > 0:
            return recommendations[:3]
        
        return None
        
    except Exception as e:
        logging.error(f"Error in generate_openai_recommendations: {e}")
        return None

def get_smart_fallback_recommendations(answers):
    """Generate smart recommendations based on user answers without AI"""
    text = " ".join(str(v).lower() for v in answers.values())
    
    # Basic keyword matching
    recommendations = []
    
    if any(word in text for word in ['anxious', 'anxiety', 'panic', 'worried', 'nervous']):
        recommendations.append({
            "title": "Managing Anxiety",
            "summary": "Evidence-based techniques for managing anxiety and panic symptoms.",
            "link": "https://www.nimh.nih.gov/health/topics/anxiety-disorders"
        })
    
    if any(word in text for word in ['depressed', 'depression', 'sad', 'hopeless', 'down']):
        recommendations.append({
            "title": "Understanding Depression", 
            "summary": "Information about depression symptoms and treatment options.",
            "link": "https://www.nimh.nih.gov/health/topics/depression"
        })
    
    if any(word in text for word in ['sleep', 'insomnia', 'tired', 'exhausted']):
        recommendations.append({
            "title": "Better Sleep Hygiene",
            "summary": "Tips for improving sleep quality and addressing sleep disorders.",
            "link": "https://www.sleepfoundation.org/sleep-hygiene"
        })
    
    if any(word in text for word in ['stress', 'overwhelmed', 'pressure', 'burnout']):
        recommendations.append({
            "title": "Stress Management",
            "summary": "Healthy ways to cope with and reduce stress in daily life.",
            "link": "https://www.apa.org/topics/stress"
        })
    
    # Fill remaining slots with general resources
    while len(recommendations) < 3:
        general_resources = [
            {
                "title": "Mental Health First Aid",
                "summary": "Basic mental health resources and when to seek professional help.",
                "link": "https://www.mentalhealth.gov/"
            },
            {
                "title": "Crisis Support Resources",
                "summary": "24/7 mental health crisis support and prevention resources.",
                "link": "https://988lifeline.org/"
            },
            {
                "title": "Mindfulness and Meditation",
                "summary": "Introduction to mindfulness practices for mental wellbeing.",
                "link": "https://www.mindful.org/meditation/mindfulness-getting-started/"
            }
        ]
        
        for resource in general_resources:
            if resource not in recommendations and len(recommendations) < 3:
                recommendations.append(resource)
    
    return recommendations[:3]

def get_fallback_recommendations():
    """Get basic fallback recommendations when everything fails"""
    return [
        {
            "title": "Mental Health Resources",
            "summary": "Comprehensive mental health information and resources from the U.S. government.",
            "link": "https://www.mentalhealth.gov/"
        },
        {
            "title": "Crisis Support - 988 Lifeline", 
            "summary": "24/7 crisis support and suicide prevention resources available nationwide.",
            "link": "https://988lifeline.org/"
        },
        {
            "title": "Anxiety and Depression Help",
            "summary": "Evidence-based resources for managing anxiety and depression symptoms.",
            "link": "https://adaa.org/understanding-anxiety"
        }
    ]

# Topic generation endpoint using OpenAI
@application.route("/get-topic", methods=["OPTIONS", "POST"])
def get_topic():
    """Generate topic title for chat"""
    if request.method == "OPTIONS":
        return cors_response({}, 204)

    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
        logging.info(f"get-topic called with text length: {len(text)}")

        if not text:
            return cors_response({"topic": "Mental Health Chat"})

        def clean_and_correct_text(input_text):
            """Clean and correct common typos"""
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
            """Capitalize every word in title"""
            return " ".join(word.capitalize() for word in title_str.split())

        cleaned_text = clean_and_correct_text(text)
        
        # Try OpenAI first, then fall back to keyword-based generation
        try:
            title = generate_openai_title(cleaned_text)
            if title and 3 <= len(title) <= 45:
                return cors_response({"topic": capitalize_every_word(title)})
        except Exception as e:
            logging.warning(f"OpenAI title generation failed: {e}")
        
        # Fallback to keyword-based title generation
        title = generate_title_from_keywords(cleaned_text)
        return cors_response({"topic": title})

    except Exception as e:
        logging.error(f"Unexpected error in /get-topic: {e}")
        return cors_response({"topic": "Mental Health Chat"})

def generate_openai_title(text):
    """Generate title using OpenAI"""
    try:
        prompt = f"""Generate a precise 2-3 word title that summarizes the core mental health topic from this message.

Requirements:
- Use warm, supportive language
- Focus on help/support rather than problems
- Be concise but meaningful (2-3 words only)
- Avoid medical jargon

Examples:
"I keep having panic attacks" → "Anxiety Support"
"My depression is getting worse" → "Mood Support"  
"Can't sleep, mind racing" → "Sleep Help"
"Relationship problems" → "Relationship Support"
"Feeling overwhelmed" → "Stress Management"

Input: "{text[:200]}"

Response (title only, no quotes or extra text):"""

        response = llm.invoke([{"role": "user", "content": prompt}])
        title = response.content.strip()
        
        # Clean the title
        title = title.strip('"\'.,!?:').strip()
        
        # Remove common prefixes
        prefixes = ['title:', 'topic:', 'response:', 'answer:']
        for prefix in prefixes:
            if title.lower().startswith(prefix):
                title = title[len(prefix):].strip()
                break
        
        # Validate title
        if title and not title.lower().startswith('i ') and len(title.split()) <= 4:
            return title
        
        return None
        
    except Exception as e:
        logging.error(f"Error in generate_openai_title: {e}")
        return None

def generate_title_from_keywords(text_lower):
    """Generate title based on keywords without AI"""
    text_lower = text_lower.lower()
    
    # Priority keywords for specific topics
    if any(term in text_lower for term in ['panic attack', 'panic disorder']):
        return "Panic Support"
    elif any(term in text_lower for term in ['social anxiety', 'social phobia']):
        return "Social Anxiety Help"
    elif any(term in text_lower for term in ['anxiety', 'anxious', 'worried', 'nervous']):
        return "Anxiety Support"
    elif any(term in text_lower for term in ['major depression', 'clinical depression']):
        return "Depression Care"
    elif any(term in text_lower for term in ['depression', 'depressed', 'sad', 'hopeless', 'down']):
        return "Mood Support"
    elif any(term in text_lower for term in ['burnout', 'work stress', 'job stress']):
        return "Work Stress Help"
    elif any(term in text_lower for term in ['stress', 'overwhelmed', 'pressure']):
        return "Stress Management"
    elif any(term in text_lower for term in ['ptsd', 'trauma', 'traumatic']):
        return "Trauma Recovery"
    elif any(term in text_lower for term in ['grief', 'grieving', 'loss', 'bereavement']):
        return "Grief Support"
    elif any(term in text_lower for term in ['insomnia', 'sleep problem', 'can\'t sleep']):
        return "Sleep Support"
    elif any(term in text_lower for term in ['relationship', 'partner', 'marriage', 'dating']):
        return "Relationship Help"
    elif any(term in text_lower for term in ['family', 'parents', 'children']):
        return "Family Support"
    elif any(term in text_lower for term in ['addiction', 'substance', 'alcohol', 'drugs']):
        return "Addiction Recovery"
    elif any(term in text_lower for term in ['eating disorder', 'body image']):
        return "Eating Support"
    elif any(term in text_lower for term in ['self esteem', 'confidence', 'self worth']):
        return "Confidence Building"
    elif any(term in text_lower for term in ['meditation', 'mindfulness', 'breathing']):
        return "Mindfulness Training"
    elif any(term in text_lower for term in ['therapy', 'therapist', 'counseling']):
        return "Therapy Questions"
    elif any(term in text_lower for term in ['medication', 'antidepressant', 'prescription']):
        return "Medication Support"
    else:
        return "Mental Health Support"

# Error handlers
@application.errorhandler(404)
def not_found(error):
    return cors_response({"error": "Endpoint not found"}, 404)

@application.errorhandler(500)
def internal_error(error):
    return cors_response({"error": "Internal server error"}, 500)

if __name__ == '__main__':
    logging.info("Starting Mental Health Chatbot Backend...")
    application.run(host="0.0.0.0", port=5000, debug=True)