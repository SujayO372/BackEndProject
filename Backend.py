import glob, os

from flask import Flask, request, jsonify, make_response
from dotenv import load_dotenv
from flask_cors import CORS
from typing_extensions import List, TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph

# TODO: Create a file called .env, then paste the following line into it:
# OPENAI_API_KEY=<your api key> (without the <>)
# The reason this did not get uploaded is because I put '.env' into the .gitignore file
# This makes it so that the .env file, which has sensitive information like your API keys,
# will not be added to the stage when you commit and thus not be added to github when you push
# Once you do this, the following lines will check that your .env exists and that your API key
# is inside it before proceeding:

load_dotenv(dotenv_path='.env')

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set")

# TODO: Write one line of code to wrap your application in CORS
# https://flask-cors.readthedocs.io/en/latest/api.html
# Hint: 'application' corresponds to the app parameter in the CORS constructor
application = Flask(__name__)
# CORS(...)

# TODO: Initialize llm using ChatOpenAI constructor
# https://python.langchain.com/docs/integrations/chat/openai/

# llm = ...;

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)
vector_store = InMemoryVectorStore(embeddings)

def load_documents():
    pdf_files = glob.glob("documents/*.pdf")
    docs = []

    # TODO: Figure out how to load docs using the LangChain Documentation locally using
    # document loaders - https://python.langchain.com/docs/integrations/document_loaders/
    # Our goal here is to load documents (for now, just download PDFs) into a variable called 'docs'
    # which we will use in our chatbot.

    # Consider using the PyPDF loader (which requires you to store PDFs locally)
    # This means you will need to upload PDF files related to mental health in the 'documents' directory.
    # In a later iteration we will upload these docs to Amazon S3, an online database, so that users
    # that download your app can still use the chatbot without having to connect to your local PC and
    # access them in this 'documents' directory.

    # Once you figure out how to load the documents, assign them to the variable 'docs'
    # The following steps are taken care of for you - go over them so you get an understanding
    # of what is happening.

    for pdf_path in pdf_files:
        # YOUR CODE HERE (replace 'pass' with your code) - Intended answer is 3 lines of code
        pass

    # This splits your documents into chunks of text

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    all_splits = text_splitter.split_documents(docs)

    # This adds your text chunks to the vector store
    vector_store.add_documents(all_splits)

load_documents()

# This initializes the original prompt for the chatbot

prompt_template = """
You are a helpful mental health assistant. Use the following context to answer the user's question.

Context:
{context}

Question: {question}

Answer:
""".strip()

# Classes and methods for retrieval, generation, and building a graph of responses

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])

    # FUTURE ITERATIONS: Comment this out before deploy
    for i in range(len(retrieved_docs)):
        print("doc", i)
        print(retrieved_docs[i])
        print()
    ################################

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

fallback_message = (
    "I apologize, but I don't have enough information in my knowledge base to answer that question accurately. "
    "Please try asking something else, or consult additional resources."
)

# Your flask application - write the code for this and paste code from doc

# Endpoint 1: query
@application.route('/query', methods=['OPTIONS', 'POST'])
def query():
    frontend_url = '';# TODO: Set this to the localhost URL of your frontend
    
    # PREFLIGHT Request: checks that the frontend is "correct"
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = frontend_url
        # TODO: Set the rest of the appropriate headers (see doc)
        return response, 204
    
    # Actual Request: send data to the frontend
    data = request.json
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400
    final_state = graph.invoke({"question": user_query})
    answer = final_state["answer"].strip()
    
    # Build and return the response
    response = jsonify({
        "response": {
            "query": user_query,
            "result": answer
        }
    })
    response.headers['Access-Control-Allow-Origin'] = frontend_url
    # TODO: Set the rest of the appropriate headers (see doc)
    return response

# Endpoint 2: refresh (FUTURE ITERATIONS)
@application.route('/refresh', methods=['POST'])
def refresh_data():
    try:
        vector_store.clear()

        # TODO: Load new data from Amazon S3
        return jsonify({"status": "success", "message": "PDFs reloaded"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Run the application
if __name__ == '__main__':
    application.run(debug=True)

