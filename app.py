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

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()

# Store conversations in memory
conversations = {}

# OpenRouter API Key - Replace with environment variable in production
OPENROUTER_API_KEY = "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"

# Similarity threshold for matching questions
SIMILARITY_THRESHOLD = 0.3

# Text preprocessing function
def cleanup(sentence):
    """Clean and lemmatize input text for better matching"""
    sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)
    words = nltk.word_tokenize(sentence.lower())
    lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(lemmatized_words)

# Load FAQ data
try:
    data = pd.read_csv('hklFAQsv2.csv')
    logging.info("FAQ data loaded successfully")
except FileNotFoundError:
    logging.error("FAQ data file not found")
    data = pd.DataFrame(columns=['Question', 'Answer', 'Class'])

# Load navigation data
try:
    navigation_data = pd.read_csv('hklNavigation.csv')
    logging.info("Navigation data loaded successfully")
except FileNotFoundError:
    logging.error("Navigation data file not found")
    navigation_data = pd.DataFrame(columns=['Building', 'Level', 'Location', 'Instructions'])

# Initialize TF-IDF vectorizer and model if data is available
if not data.empty:
    questions = data['Question'].values
    tfv = TfidfVectorizer(min_df=1, stop_words='english', ngram_range=(1, 2))
    
    # Preprocess questions
    X = [cleanup(question) for question in questions]
    tfv.fit(X)
    X = tfv.transform(X)
    
    # Train classification model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, data['Class'])
    logging.info("Model trained successfully")
else:
    model = None
    tfv = None
    logging.warning("No data available for model training")

def call_openrouter_api(user_message):
    """Call OpenRouter API for queries that don't match existing FAQs"""
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
            logging.error(f"OpenRouter API Error: {response.status_code}, {response.text}")
            return "Sorry, I'm having trouble fetching an answer right now."
    except Exception as e:
        logging.error(f"Exception during OpenRouter API call: {str(e)}")
        return "Sorry, I couldn't get an answer at the moment."

# Routes
@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    """Initialize a new conversation"""
    conversation_id = len(conversations) + 1
    conversations[conversation_id] = []
    return jsonify({"conversation_id": conversation_id})

@app.route('/ask/<int:conversation_id>', methods=['POST'])
def ask(conversation_id):
    """Process user queries and return appropriate responses"""
    user_input = request.json.get('message', '')
    if not user_input:
        return jsonify({"response": "No input provided."}), 400

    cleaned_input = cleanup(user_input)

    # Check for navigation info
    nav_info = navigation_data[navigation_data['Location'].str.contains(cleaned_input, case=False, na=False)]
    if not nav_info.empty:
        response_lines = [
            f"<b>Location: </b>{nav_info.iloc[0]['Location']}",
            f"<b>Building: </b>{nav_info.iloc[0]['Building']}",
            f"<b>Level: </b>{nav_info.iloc[0]['Level']}",
            f"<b>Instructions: </b>{nav_info.iloc[0]['Instructions']}"
        ]
        suggestions = navigation_data[navigation_data['Location'].str.contains(cleaned_input, case=False, na=False)]['Location'].tolist()[:4]
        conversations[conversation_id].append({"user": user_input, "bot": response_lines})
        return jsonify({"response": response_lines, "suggestions": suggestions})

    # Check FAQ matching
    if model is None or tfv is None:
        return jsonify({"response": "Model not trained. Please check the data."}), 500

    # Calculate similarity with existing questions
    input_vec = tfv.transform([cleaned_input])
    all_questions_vec = tfv.transform([cleanup(q) for q in data['Question']])
    sims = cosine_similarity(input_vec, all_questions_vec).flatten()

    # Get the top 4 suggestions based on similarity scores
    top_indices = np.argsort(sims)[-4:][::-1]  # Get indices of top 4 suggestions
    top_scores = sims[top_indices]

    # Prepare FAQ suggestions
    faq_suggestions = []
    for idx, score in zip(top_indices, top_scores):
        # Always include top suggestions
        faq_suggestions.append(data.iloc[idx]['Question'])

    # Decide whether to use FAQ or call API based on similarity
    if any(score >= SIMILARITY_THRESHOLD for score in top_scores):
        answer = data.iloc[top_indices[0]]['Answer']  # Get the best matching FAQ answer
    else:
        answer = call_openrouter_api(user_input)

    conversations[conversation_id].append({"user": user_input, "bot": answer})
    return jsonify({"response": answer, "suggestions": faq_suggestions})

@app.route('/faq_classes', methods=['GET'])
def faq_classes():
    """Return all FAQ categories"""
    classes = data['Class'].unique().tolist()
    return jsonify({"classes": classes})

@app.route('/faqs/<class_name>', methods=['GET'])
def get_faqs(class_name):
    """Return FAQs for a specific category"""
    faqs = data[data['Class'] == class_name][['Question', 'Answer']].to_dict(orient='records')
    return jsonify({"faqs": faqs})

@app.route('/history/<int:conversation_id>', methods=['GET'])
def get_history(conversation_id):
    """Return conversation history"""
    if conversation_id not in conversations:
        return jsonify({"error": "Conversation not found."}), 404
    return jsonify({"history": conversations[conversation_id]})

@app.route('/navigation/<location>', methods=['GET'])
def get_navigation(location):
    """Return navigation information for a specific location"""
    nav_info = navigation_data[navigation_data['Location'].str.contains(location, case=False, na=False)]
    
    if nav_info.empty:
        return jsonify({"message": "Location not found."}), 404
    
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
    