import glob, os

from flask import Flask, request, jsonify, make_response
from dotenv import load_dotenv
from flask_cors import CORS
from typing_extensions import List, TypedDict

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph

# Load environment variables
load_dotenv(dotenv_path='.env')

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set")

# Flask application
application = Flask(__name__)
CORS(application)  # ✅ Enable CORS

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4",  # or "gpt-3.5-turbo"
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

# Vector store setup
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

vector_store = InMemoryVectorStore(embeddings)

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

# Prompt for chatbot
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
    for i in range(len(retrieved_docs)):
        print("doc", i)
        print(retrieved_docs[i])
        print()
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

# fallback_message = (
#     "I apologize, but I don't have enough information in my knowledge base to answer that question accurately. "
#     "Please try asking something else, or consult additional resources."
# )

# ✅ Replace this with your actual frontend URL
frontend_url = 'http://localhost:3000'

# Endpoint 1: /query
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

    final_state = graph.invoke({"question": user_query})
    answer = final_state["answer"].strip()

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

# Endpoint 2: /refresh
@application.route('/refresh', methods=['POST'])
def refresh_data():
    try:
        vector_store.clear()
        # TODO: Load new data from Amazon S3
        return jsonify({"status": "success", "message": "Text files reloaded"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ✅ New Endpoint 3: /gemini-query
@application.route('/gemini-query', methods=['OPTIONS', 'POST'])
def gemini_query():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = frontend_url
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response, 204

    from google import genai
    from google.genai import types

    data = request.json
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400

    try:
        client = genai.Client()

        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        config = types.GenerateContentConfig(
            tools=[grounding_tool]
        )

        gemini_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_query,
            config=config,
        )

        candidate = gemini_response.candidates[0]
        answer_text = candidate.content.parts[0].text

        # Extract grounded web sources
        sources = []
        chunks = candidate.groundingMetadata.groundingChunks
        for chunk in chunks:
            web = chunk.get("web")
            if web:
                sources.append({
                    "url": web.get("uri"),
                    "title": web.get("title")
                })

        response = jsonify({
            "response": {
                "query": user_query,
                "result": answer_text,
                "sources": sources
            }
        })
        response.headers['Access-Control-Allow-Origin'] = frontend_url
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    application.run(debug=True)
