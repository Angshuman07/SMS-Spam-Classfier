from flask import Flask, render_template, request, jsonify
import joblib, re
from pathlib import Path

app = Flask(__name__)

vect = joblib.load("models/tfidf_vectorizer.joblib")
clf = joblib.load("models/multinomial_nb.joblib")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return " ".join(text.split())

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    msg = request.form.get("message") or request.json.get("message")
    if not msg:
        return jsonify({"error": "No message provided"}), 400
    X = vect.transform([preprocess(msg)])
    proba = clf.predict_proba(X)[0]
    label = "spam" if proba[1] > 0.5 else "ham"
    return jsonify({"label": label, "probability": float(proba[1])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
