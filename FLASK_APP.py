import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # Enable CORS for frontend integration
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define working directory
WORKING_DIR = "./legisys"

# Initialize LightRAG
rag = LightRAG(
    working_dir=WORKING_DIR,
    embedding_func=openai_embed,
    llm_model_func=gpt_4o_mini_complete  # GPT-4o Mini model
)

# Default Query Parameters (mix = KG + VectorDB)
query_param = QueryParam(mode="mix")

# Custom Dutch prompt for legal assistant
custom_prompt = """
Je bent een juridisch assistent die alle vragen van de advocaat moet beantwoorden in de Nederlandse taal, 
met gebruik van de context en juridische informatie die je is gegeven (ook in de Nederlandse taal).
"""

# Function to process queries
def process_query(query):
    if not query:
        return {"error": "Query parameter is required"}, 400

    response = rag.query(query, param=query_param, prompt=custom_prompt)
    return {"query": query, "response": response}

# Endpoint: /caseanalysis
@app.route("/caseanalysis", methods=["POST"])
def caseanalysis():
    data = request.json
    query = data.get("query")
    return jsonify(process_query(query))

# Endpoint: /ragquery
@app.route("/ragquery", methods=["POST"])
def ragquery():
    data = request.json
    query = data.get("query")
    return jsonify(process_query(query))

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
