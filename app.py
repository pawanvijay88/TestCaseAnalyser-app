from flask import Flask, request, render_template
import pickle

# Load the Multinomial Naive Bayes model and TfidfVectorizer object from disk
tfidf = pickle.load(open('tfidf-transform.pkl', 'rb'))
classifier = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf.transform(data).toarray()
        print(vect)
        my_prediction = classifier.predict(vect)
        print(my_prediction)
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)