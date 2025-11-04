# SMS Spam Classifier (TF-IDF + MultinomialNB)

### Features
- NLP-based spam detection using TF-IDF and MultinomialNB.
- Preprocessing via tokenization, stopword removal, and stemming.
- Flask web interface for real-time message classification.

### Usage
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train.py --data data/sms_small.csv

# 4. Run Flask app
python app.py
