from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from evaluate import load

app = Flask(__name__)

# Load models
print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load evaluation metrics
rouge = load("rouge")
bleu = load("bleu")

# Knowledge base (documents and references)
documents = [
    "Solar energy reduces electricity bills and is renewable.",
    "Wind energy is a renewable source of power.",
    "Renewable energy helps combat climate change."
]
references = documents  # Use documents as references for evaluation

# Load FAISS index for retrieval
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
    print(f"Query: {query}\nRetrieved Documents: {retrieved_docs}\n", flush=True)
    return retrieved_docs

# Generate response with retrieval (RAG)
def generate_with_rag(query):
    retrieved_docs = retrieve_context(query, k=1)
    context = " ".join(retrieved_docs)
    formatted_input = f"{query} Context: {context}"
    inputs = tokenizer(formatted_input, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate response without retrieval (Baseline)
def generate_without_rag(query):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluate the system
def evaluate_system(queries):
    rag_predictions = []
    baseline_predictions = []

    # Generate responses
    for query in queries:
        rag_predictions.append(generate_with_rag(query))
        baseline_predictions.append(generate_without_rag(query))

    # Compute metrics
    rag_metrics = {
        "rouge": rouge.compute(predictions=rag_predictions, references=references),
        "bleu": bleu.compute(predictions=rag_predictions, references=references),
    }
    baseline_metrics = {
        "rouge": rouge.compute(predictions=baseline_predictions, references=references),
        "bleu": bleu.compute(predictions=baseline_predictions, references=references),
    }

    # Combine results
    results = {
        "rag": {
            "predictions": rag_predictions,
            "metrics": rag_metrics,
        },
        "baseline": {
            "predictions": baseline_predictions,
            "metrics": baseline_metrics,
        }
    }
    return results

# Define /evaluate endpoint
@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    queries = data.get("queries", [])

    if not queries:
        return jsonify({"error": "Queries are required"}), 400

    results = evaluate_system(queries)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
