from flask import Flask, render_template_string, request
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SMS Spam Classifier</title>
    <style>
        body {font-family: Arial; background-color: #f7f8fa; text-align: center; margin-top: 100px;}
        form {background-color: white; display: inline-block; padding: 30px; border-radius: 15px; box-shadow: 0px 0px 15px #ccc;}
        input[type=text] {width: 300px; padding: 10px; margin: 10px;}
        input[type=submit] {background-color: #4CAF50; color: white; border: none; padding: 10px 20px; cursor: pointer;}
        .result {font-size: 18px; margin-top: 20px;}
    </style>
</head>
<body>
    <h1>📩 SMS Spam Classifier</h1>
    <form method="post">
        <input type="text" name="message" placeholder="Enter your message here" required />
        <br>
        <input type="submit" value="Predict" />
    </form>
    {% if prediction %}
    <div class="result">
        <p><b>Prediction:</b> {{ prediction }}</p>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        message = request.form['message']
        transformed_msg = vectorizer.transform([message])
        pred = model.predict(transformed_msg)[0]
        prediction = "🚫 Spam" if pred == 'spam' else "✅ Not Spam"
    return render_template_string(HTML, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
