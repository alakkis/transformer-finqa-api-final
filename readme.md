# FinQA Transformer API (Flask + Docker)

## Overview

This project builds an end-to-end financial question-answering system using a fine-tuned Transformer model (T5) trained on the FinQA dataset. It exposes the model through a Flask-based REST API, runs it inside a Docker container, and tests it via Postman. The system is deployable locally or to the cloud with minimal changes. While this implementation currently runs on CPU, it can be adapted to GPU environments for improved performance.

## Project Structure

```
finqa-flask-api/
├── app.py                  # Flask app to serve model predictions
├── Dockerfile              # Instructions to containerize the app
├── requirements.txt        # Python dependencies
├── transformer-finqa.ipynb # Notebook: data prep, model, evaluation
├── model/                  # Local tokenizer and model files
│   ├── config.json
│   ├── spiece.model
│   └── model.safetensors
...
```

## How the System Works

1. A financial question is submitted via the API.
2. The Flask app processes the request and tokenizes the question and table.
3. The pre-trained T5 model generates a text-based answer.
4. The answer is returned to the user in JSON format.

The model and tokenizer are loaded from disk (`model/` folder) for faster inference and offline portability.

---

## Core Technologies

| Component        | Technology       | Purpose                                                    |
| ---------------- | ---------------- | ---------------------------------------------------------- |
| Model            | T5 Transformer   | Fine-tuned to answer finance questions from tables (FinQA) |
| Framework        | Flask            | Exposes the model via a RESTful API endpoint               |
| Environment      | Docker           | Containerizes the API for deployment                       |
| Tokenization     | SentencePiece    | Subword tokenization for T5 input format                   |
| Inference Engine | PyTorch          | Powers the Transformer inference                           |
| Testing Tool     | Postman          | Sends API requests and receives predictions                |
| CLI Tool         | Terminal (macOS) | Controls Docker, Flask, and scripts                        |

These technologies form a coherent pipeline: Docker builds and runs the app container, Flask hosts the API endpoint, PyTorch runs the model inside the container, and Postman interfaces with the API externally.

---

## Deployment Readiness

This system is designed to be deployable:

* Locally via `python3 app.py`
* Inside Docker via:

  ```bash
  docker build -t finqa-api .
  docker run -p 5001:5001 finqa-api
  ```
* To production with minimal changes (e.g. Render, Heroku, AWS)

After deployment, you can query the model with a POST request to `/predict`.

---

## Example Usage (Postman)

**POST** `http://127.0.0.1:5001/predict`

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

* Build a real-world ML product from notebook to API
* Practice MLOps principles (isolation, testing, versioning)
* Learn how AI, systems, and deployment interact
* Demonstrate awareness of production-readiness and future CI/CD integration

Supporting docs will include:

* `notebook_build.md`: Full walkthrough of the notebook pipeline
* `technologies_used.md`: Deep explanation of the stack & system links
* `transformer_math.md`: Mathematical reasoning behind T5 and QA flow
* `deployment_guide.md`: Instructions for launching to the cloud

---

This repository demonstrates practical ML engineering through modern tooling and real financial language modeling. Suitable for AI engineering portfolios, MLOps learning, or FinNLP experimentation.
