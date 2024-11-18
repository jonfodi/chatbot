from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import sys


app = Flask(__name__)

# Load models
print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index and documents
documents = [
    "Solar energy reduces electricity bills and is renewable.",
    "Wind energy is a renewable source of power.",
    "Renewable energy helps combat climate change."
]
embeddings = embedding_model.encode(documents)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
doc_ids = {i: doc for i, doc in enumerate(documents)}

# Define retriever
def retrieve_context(query, k=1):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=k)
    retrieved_docs = [doc_ids[i] for i in indices[0]]  # Retrieve top-k documents
    print(f"Query: {query}\nRetrieved Documents: {retrieved_docs}\n", file=sys.stderr, flush=True)   # Log the retrieved documents

    return retrieved_docs

# Define /chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("message", "")

    # Retrieve context
    retrieved_docs = retrieve_context(query, k=1)
    context = " ".join(retrieved_docs)

    # Combine query and context
    formatted_input = f"{query} Context: {context}"

    # Generate response
    inputs = tokenizer(formatted_input, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
 