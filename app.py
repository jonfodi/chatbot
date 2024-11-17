from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer globally
print("Loading the model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
print("Model loaded successfully!")

# Define the chatbot API route
@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Parse the user's input
        data = request.get_json()
        user_input = data.get("message", "")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        # Generate response using the model
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"], max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return the response
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
