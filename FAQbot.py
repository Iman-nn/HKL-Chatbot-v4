import pandas as pd
import numpy as np
import pickle
import operator
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.metrics.pairwise import cosine_similarity
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer

# Initialize the stemmer
stemmer = LancasterStemmer()

# Function to clean up user input
def cleanup(sentence):
    word_tok = nltk.word_tokenize(sentence)
    stemmed_words = [stemmer.stem(w) for w in word_tok]
    return ' '.join(stemmed_words)

# Initialize label encoder and TF-IDF vectorizer
le = LE()
tfv = TfidfVectorizer(min_df=1, stop_words='english')

# Load data from CSV
data = pd.read_csv('hklFAQsv2.csv')
questions = data['Question'].values

# Preprocess questions
X = [cleanup(question) for question in questions]

# Fit the vectorizer and label encoder
tfv.fit(X)
le.fit(data['Class'])

# Transform the data
X = tfv.transform(X)
y = le.transform(data['Class'])

# Split the data into training and testing sets
trainx, testx, trainy, testy = tts(X, y, test_size=.25, random_state=42)

# Train the SVM model
model = SVC(kernel='linear')
model.fit(trainx, trainy)
print("SVC:", model.score(testx, testy))

# Function to get the indices of the top 5 results
def get_max5(arr):
    ixarr = [(el, ix) for ix, el in enumerate(arr)]
    ixarr.sort()
    ixs = [i[1] for i in ixarr[-5:]]
    return ixs[::-1]

# Chat function
def chat():
    cnt = 0
    print("PRESS Q to QUIT")
    print("TYPE \"DEBUG\" to Display Debugging statements.")
    print("TYPE \"STOP\" to Stop Debugging statements.")
    print("TYPE \"TOP5\" to Display 5 most relevant results")
    print("TYPE \"CONF\" to Display the most confident result")
    print()
    
    DEBUG = False
    TOP5 = False

    print("Bot: Hi, Welcome to my Chatbot!")
    while True:
        usr = input("You: ")

        if usr.lower() in ['yes', 'no']:
            print("Bot:", "Yes!" if usr.lower() == 'yes' else "No?")
            continue

        if usr == 'DEBUG':
            DEBUG = True
            print("Debugging mode on")
            continue

        if usr == 'STOP':
            DEBUG = False
            print("Debugging mode off")
            continue

        if usr.upper() == 'Q':
            print("Bot: It was good to be of help.")
            break

        if usr == 'TOP5':
            TOP5 = True
            print("Will display 5 most relevant results now")
            continue

        if usr == 'CONF':
            TOP5 = False
            print("Only the most relevant result will be displayed")
            continue

        # Transform user input
        t_usr = tfv.transform([cleanup(usr.strip().lower())])
        
        # Make prediction
        prediction = model.predict(t_usr)
        if prediction.size == 0:
            print("No prediction made. Please try again.")
            continue  # Skip the rest of the loop if no prediction is available.

        class_ = le.inverse_transform(prediction)[0]
        questionset = data[data['Class'] == class_]

        if DEBUG:
            print("Question classified under category:", class_)
            print("{} Questions belong to this class".format(len(questionset)))

        cos_sims = []
        for question in questionset['Question']:
            sims = cosine_similarity(tfv.transform([question]), t_usr)
            cos_sims.append(sims[0][0])  # Extract the scalar value

        ind = cos_sims.index(max(cos_sims))

        if DEBUG:
            question = questionset["Question"].iloc[ind]
            print("Assuming you asked: {}".format(question))

        if not TOP5:
            print("Bot:", data['Answer'].iloc[questionset.index[ind]])
        else:
            inds = get_max5(cos_sims)
            for ix in inds:
                print("Question: " + data['Question'].iloc[questionset.index[ix]])
                print("Answer: " + data['Answer'].iloc[questionset.index[ix]])
                print('-' * 50)

        print("\n" * 2)
        outcome = input("Was this answer helpful? Yes/No: ").lower().strip()
        if outcome == 'yes':
            cnt = 0
        elif outcome == 'no':
            inds = get_max5(cos_sims)
            sugg_choice = input("Bot: Do you want me to suggest you questions? Yes/No: ").lower()
            if sugg_choice == 'yes':
                q_cnt = 1
                for ix in inds:
                    print(q_cnt, "Question: " + data['Question'].iloc[questionset.index[ix]])
                    q_cnt += 1
                num = int(input("Please enter the question number you find most relevant: "))
                print("Bot: ", data['Answer'].iloc[questionset.index[inds[num - 1]]])

# Start the chat
chat()