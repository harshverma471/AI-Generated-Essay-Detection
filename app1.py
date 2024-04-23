from flask import Flask, render_template, request
import joblib
import time

app = Flask(__name__)

clf, tokenizer, vectorizer1 = joblib.load('model_preprocessed.pkl')

def preprocess_text(text):
    tokenized_text = tokenizer.tokenize(text)
    vectorized_text = vectorizer1.transform([tokenized_text])
    return vectorized_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    essay = request.args.get('essay')
    if essay:
        preprocessed_input = preprocess_text(essay)
        prediction = clf.predict(preprocessed_input)
        result = "The essay is predicted to be written by AI." if prediction == 1 else "The essay is predicted to be written by a human."

        time.sleep(2)
        return result
    else:
        return 'Error: No essay provided'

if __name__ == '__main__':
    app.run(debug=True)
