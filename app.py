from flask import Flask, request, jsonify, render_template
import joblib
import requests

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('nb_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# News API Key (replace with your valid NewsAPI key)
NEWS_API_KEY = '2ea98611df044a1a8df1147b1d0ab2b7'
NEWS_API_URL = 'https://newsapi.org/v2/top-headlines?category=business&country=us&'

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        statement = request.form['news']  # Updated to match the HTML form's 'name' attribute
        processed_statement = preprocess_text(statement)
        vectorized_statement = vectorizer.transform([processed_statement])
        prediction = model.predict(vectorized_statement)
        return render_template('index.html', prediction=prediction[0], statement=statement)

@app.route('/business-news', methods=['GET'])
def business_news():
    # Fetch business news from NewsAPI
    params = {
        'category': 'business',
        'country': 'us',  # Change to your desired country code
        'apiKey': NEWS_API_KEY
    }

    try:
        response = requests.get(NEWS_API_URL, params=params)
        print("API Request URL:", response.url)  # This will print the URL being used to fetch news

        # Check if the response was successful (status code 200)
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            return render_template('business_news.html', articles=[], error="Unable to fetch news at the moment.")

        news_data = response.json()

        # Check if the API response contains articles
        if news_data.get('status') == 'ok' and news_data.get('articles'):
            articles = news_data.get('articles', [])
        else:
            articles = []
            error_message = "No articles found."
            return render_template('business_news.html', articles=articles, error=error_message)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return render_template('business_news.html', articles=[], error="An error occurred while fetching the news.")

    return render_template('business_news.html', articles=articles)

def preprocess_text(text):
    text = text.lower()
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token.isalnum()]
    processed_text = ' '.join(filtered_tokens)
    return processed_text

if __name__ == '__main__':
    app.run(port=5001, debug=True)  # Enable debug mode for easier troubleshooting
