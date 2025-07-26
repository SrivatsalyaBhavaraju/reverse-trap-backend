from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)



@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    msg = data.get("message", "")
    result = classifier(msg)
    top = result[0]
    emotion = top['label']
    score = round(top['score'] * 100)
    manipulation_type = "guilt" if emotion in ["sadness", "fear"] else "flattery" if emotion == "joy" else "urgency" if emotion == "anger" else "neutral"

    safe_reply = {
        "guilt": "I’m not comfortable with this conversation right now.",
        "urgency": "Let’s pause and come back to this later.",
        "flattery": "Thanks, but let’s keep things clear.",
        "neutral": "Okay!"
    }[manipulation_type]

    return jsonify({
        "emotion": emotion,
        "confidence": score,
        "manipulation_type": manipulation_type,
        "safe_reply": safe_reply,
        "risk_score": 80 if emotion in ["anger", "fear", "sadness"] else 20
    })

# Render needs this:
port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port)
