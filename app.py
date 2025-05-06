from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import logging
import requests
import json
import re

app = Flask(__name__)
CORS(app)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Store conversations in memory
conversations = {}

# Your OpenRouter API Key (Replace with your real key)
OPENROUTER_API_KEY = "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"

# Basic cleanup function
def cleanup(sentence):
    sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)
    words = nltk.word_tokenize(sentence.lower())
    lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(lemmatized_words)

# Load FAQs data
try:
    data = pd.read_csv('hklFAQsv2.csv')
except FileNotFoundError:
    logging.error("FAQ data file not found.")
    data = pd.DataFrame(columns=['Question', 'Answer', 'Class'])

# Load navigation data
try:
    navigation_data = pd.read_csv('hklNavigation.csv')
except FileNotFoundError:
    logging.error("Navigation data file not found.")
    navigation_data = pd.DataFrame(columns=['Building', 'Level', 'Location', 'Instructions'])

# Initialize TF-IDF vectorizer and model if data is available
if not data.empty:
    questions = data['Question'].values
    tfv = TfidfVectorizer(min_df=1, stop_words='english', ngram_range=(1, 2))
    X = [cleanup(question) for question in questions]
    tfv.fit(X)
    X = tfv.transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, data['Class'])
else:
    model = None
    tfv = None

def call_openrouter_api(user_message):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "qwen/qwen3-1.7b:free",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant answering ONLY about Hospital Kuala Lumpur (HKL). If the question is unrelated to HKL, politely say you cannot answer."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    }
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )
        if response.status_code == 200:
            reply = response.json()['choices'][0]['message']['content']
            return reply
        else:
            print(f"OpenRouter API Error: {response.status_code}, {response.text}")
            return "Sorry, I'm having trouble fetching a smart answer right now."
    except Exception as e:
        print(f"Exception during OpenRouter API call: {str(e)}")
        return "Sorry, I couldn't get an answer at the moment."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    conversation_id = len(conversations) + 1
    conversations[conversation_id] = []
    return jsonify({"conversation_id": conversation_id})

@app.route('/ask/<int:conversation_id>', methods=['POST'])
def ask(conversation_id):
    user_input = request.json.get('message', '')
    if not user_input:
        return jsonify({"response": "No input provided."}), 400

    cleaned_input = cleanup(user_input)

    # Check for navigation keywords
    nav_info = navigation_data[navigation_data['Location'].str.contains(cleaned_input, case=False, na=False)]
    
    if not nav_info.empty:
        # Create a response message as a list of lines
        response_lines = [
            nav_info.iloc[0]['Location'],
            nav_info.iloc[0]['Building'],
            f"Level: {nav_info.iloc[0]['Level']}",
            f"Instructions: {nav_info.iloc[0]['Instructions']}"
        ]
        conversations[conversation_id].append({"user": user_input, "bot": response_lines})
        return jsonify({"response": response_lines})

    # If no navigation found, suggest similar locations
    similar_locations = navigation_data[navigation_data['Location'].str.contains(cleaned_input, case=False, na=False)]
    if not similar_locations.empty:
        suggestions = similar_locations['Location'].tolist()
        return jsonify({"response": "Did you mean:", "suggestions": suggestions})

    # If no FAQs or navigation found, fallback to FAQ
    if model is None or tfv is None:
        return jsonify({"response": "Model not trained. Please check the data."}), 500

    input_vec = tfv.transform([cleaned_input])
    all_questions_vec = tfv.transform([cleanup(q) for q in data['Question']])
    sims = cosine_similarity(input_vec, all_questions_vec).flatten()

    best_idx = np.argmax(sims)
    best_score = sims[best_idx]

    SIMILARITY_THRESHOLD = 0.3

    if best_score >= SIMILARITY_THRESHOLD:
        answer = data.iloc[best_idx]['Answer']
    else:
        answer = call_openrouter_api(user_input)

    conversations[conversation_id].append({"user": user_input, "bot": answer})
    return jsonify({"response": answer})

@app.route('/faq_classes', methods=['GET'])
def faq_classes():
    classes = data['Class'].unique().tolist()
    return jsonify({"classes": classes})

@app.route('/faqs/<class_name>', methods=['GET'])
def get_faqs(class_name):
    faqs = data[data['Class'] == class_name][['Question', 'Answer']].to_dict(orient='records')
    return jsonify({"faqs": faqs})

@app.route('/history/<int:conversation_id>', methods=['GET'])
def get_history(conversation_id):
    if conversation_id not in conversations:
        return jsonify({"error": "Conversation not found."}), 404
    return jsonify({"history": conversations[conversation_id]})

@app.route('/navigation/<location>', methods=['GET'])
def get_navigation(location):
    nav_info = navigation_data[navigation_data['Location'].str.contains(location, case=False, na=False)]
    
    if nav_info.empty:
        return jsonify({"message": "Location not found."}), 404
    
    # Prepare a detailed response
    response_data = []
    for _, row in nav_info.iterrows():
        response_data.append({
            "Building": row['Building'],
            "Level": row['Level'],
            "Location": row['Location'],
            "Instructions": row['Instructions']
        })
    
    return jsonify({"navigation": response_data})

if __name__ == "__main__":
    app.run(debug=True)