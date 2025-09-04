# FinQA Transformer API – End-to-End System

## Overview

This project demonstrates a full-stack machine learning application: a Transformer-based question-answering system trained on the FinQA dataset and served via a REST API. Starting with dataset preparation and model fine-tuning in a Jupyter notebook, I deployed the system using Flask, Docker, Postman, and AWS EC2.

The model answers natural-language finance questions based on structured tables and is accessible via a live API.

---

## Project Timeline

### 1. Dataset Preparation
- Acquired and cleaned the FinQA dataset.
- Converted tabular data and related questions into input format compatible with the model.

### 2. Model Fine-Tuning
- Used Hugging Face `T5-base` and the FinQA dataset.
- Tokenized inputs using SentencePiece.
- Fine-tuned the model using Hugging Face's Trainer API (PyTorch backend).
- Saved model and tokenizer outputs to `model/`.

### 3. Flask API Setup
- Built a lightweight REST API in `app.py`.
- Exposed a `/predict` endpoint that accepts POST requests with JSON input:
  ```json
  {
    "question": "What is the net income in 2020?",
    "table": [["Year", "Revenue", "Net Income"], ["2020", "$120M", "$25M"]]
  }
  ```
- Returns a JSON response:
  ```json
  {
    "answer": "$25 million"
  }
  ```

### 4. Docker Integration
- Created a `Dockerfile` to containerize the app and dependencies.
- Commands used:
  ```bash
  docker build -t finqa-api .
  docker run -p 5000:5000 finqa-api
  ```

### 5. API Testing with Postman
- Sent POST requests to `http://127.0.0.1:5000/predict`.
- Validated API response structure and model accuracy.
- Simulated external use cases and confirmed robustness.

### 6. Cloud Deployment with AWS EC2
- Deployed the containerized app on a cloud-hosted Ubuntu EC2 instance.
- SSH access:
  ```bash
  ssh -i "key.pem" ubuntu@<EC2-IP>
  ```
- Installed Docker, cloned repo, and ran the container.
- Opened port 5000 for public access via AWS Security Groups.

---

## How the System Works

1. A client sends a question and table as JSON to the `/predict` endpoint.
2. Flask parses the request, tokenizes the input.
3. The T5 model generates an answer.
4. The answer is returned as a JSON response.

---

## Transformer Model: Architecture and Math

### Structure
- T5 is an encoder-decoder transformer.
- Input: flattened text of table + question.
- Output: natural-language answer.

### Key Components
1. **Token Embedding**:
  x_i \rightarrow e_i \in \mathbb{R}^d


2. **Self-Attention**:
   \[
   	ext{Attention}(Q, K, V) = 	ext{softmax}\left(rac{QK^T}{\sqrt{d_k}}
ight)V
   \]

3. **Loss Function (Cross-Entropy)**:
   \[
   \mathcal{L} = -\sum y_i \log(\hat{y}_i)
   \]

### Why It Works
- Self-attention highlights relevant parts of the table for each question.
- The decoder uses this context to generate accurate answers.
- Fine-tuning on FinQA aligns the model with domain-specific language and formats.

---

## Technologies Used

| Component     | Tool        | Description                      |
|---------------|-------------|----------------------------------|
| Model         | T5          | Transformer for QA               |
| Framework     | Flask       | API interface                    |
| Container     | Docker      | Deployment environment           |
| Testing Tool  | Postman     | API testing                      |
| Hosting       | AWS EC2     | Cloud deployment                 |
| Interface     | Terminal    | CLI for local and remote control |

---

## System Diagram

```plaintext
[Notebook] → Prepare Data & Fine-tune T5 model
    ↓
[Flask App] → Loads model & serves /predict
    ↓
[Docker] → Wraps Flask into container
    ↓
[Terminal] → Local & EC2 command-line control
    ↓
[Postman] → Simulates external requests
    ↓
[AWS EC2] → Cloud-hosted, public endpoint
```

---

## File Structure

```
finqa-flask-api/
├── app.py                  # Flask app
├── Dockerfile              # Docker container setup
├── requirements.txt        # Python dependencies
├── transformer-finqa.ipynb # Notebook for training
└── model/                  # Saved model/tokenizer
```

---

## Sample Usage

**POST** `http://<EC2-IP>:5000/predict`

**Request:**
```json
{
  "question": "What is the net income in 2020?",
  "table": [
    ["Year", "Revenue", "Net Income"],
    ["2019", "$100M", "$20M"],
    ["2020", "$120M", "$25M"]
  ]
}
```

**Response:**
```json
{
  "answer": "$25 million"
}
```

---

## Project Goals

- Build an end-to-end ML system deployable via API
- Practice Docker, Flask, and cloud deployment
- Apply research-level QA model to production setting
- Continue to build my expertise in AI engineering
