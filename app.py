from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Initialize Flask app
app = Flask(__name__)

# Load pretrained model and tokenizer from Hugging Face
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Helper function to process input and return answer
def generate_answer(question, table):
    table_str = " ".join([" | ".join(row) for row in table])
    input_text = f"question: {question} context: {table_str}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define /predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        question = data["question"]
        table = data["table"]
        answer = generate_answer(question, table)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
