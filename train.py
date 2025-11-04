#!/usr/bin/env python3
"""
train.py — SMS Spam Classifier (Auto-download version)
------------------------------------------------------
This script automatically downloads the UCI SMS Spam Collection dataset
if missing, preprocesses text (NLTK + regex cleanup), trains a TF-IDF
+ MultinomialNB model, and saves it to models/.
"""

import os
import re
import argparse
import pandas as pd
import requests, zipfile
from io import BytesIO
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -------------------------
# 🔹 Auto-download dataset
# -------------------------
def download_dataset_if_missing(path: str):
    """Download and convert the UCI SMS Spam dataset to CSV if not already present."""
    path = Path(path)
    if path.exists():
        print(f"✅ Dataset already exists at: {path}")
        return

    print("📥 Downloading UCI SMS Spam Collection dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    os.makedirs(path.parent, exist_ok=True)
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        z.extractall(path.parent)
    print("📦 Extracted dataset files.")

    df = pd.read_csv(path.parent / "SMSSpamCollection", sep="\t", names=["label", "text"])
    df.to_csv(path, index=False)
    print(f"✅ Converted to CSV: {path} ({len(df)} rows)")

# -------------------------
# 🔹 Text preprocessing
# -------------------------
try:
    import nltk
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords as nltk_stopwords
    nltk_available = True
except Exception:
    nltk_available = False

FALLBACK_STOPWORDS = set("""a about above after again against all am an and any are aren't as at be because been before
being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few
for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself
his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not
of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so
some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's
where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves""".split())

def preprocess_text(text, stemmer=None, stopwords=None):
    """Basic text cleaning, tokenization, stopword removal, and stemming."""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    if stemmer:
        tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# -------------------------
# 🔹 Load and train
# -------------------------
def main(args):
    data_path = Path(args.data)
    download_dataset_if_missing(data_path)

    df = pd.read_csv(data_path)
    print(f"📊 Loaded dataset with {len(df)} rows.")

    # Prepare stopwords and stemmer
    if nltk_available:
        try:
            stemmer = PorterStemmer()
            stopwords = set(nltk_stopwords.words("english"))
        except Exception:
            stemmer = PorterStemmer()
            stopwords = FALLBACK_STOPWORDS
    else:
        stemmer, stopwords = None, FALLBACK_STOPWORDS

    # Preprocess
    print("🧹 Preprocessing text...")
    df["clean"] = df["text"].apply(lambda s: preprocess_text(s, stemmer, stopwords))

    X = df["clean"].values
    y = df["label"].map(lambda s: 1 if str(s).lower().startswith("s") else 0).values

    strat = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=strat, random_state=42)

    print("⚙️  Training TF-IDF + MultinomialNB model...")
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train_t = vect.fit_transform(X_train)
    X_test_t = vect.transform(X_test)

    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train_t, y_train)

    y_pred = clf.predict(X_test_t)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")

    try:
        print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    except Exception:
        pass

    os.makedirs("models", exist_ok=True)
    joblib.dump(vect, "models/tfidf_vectorizer.joblib")
    joblib.dump(clf, "models/multinomial_nb.joblib")
    print("💾 Model artifacts saved in /models")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/SMSSpamCollection.csv", help="Path to dataset CSV")
    args = parser.parse_args()
    main(args)
