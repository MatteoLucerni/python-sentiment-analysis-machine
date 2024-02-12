import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from joblib import dump
from sklearn.neural_network import MLPClassifier

# preprocessing data
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    words = text.split()
    processed_words = []
    for word in words:
        if word not in stop_words:
            stemmed = stemmer.stem(word)
            lemmatized = lemmatizer.lemmatize(stemmed)
            processed_words.append(lemmatized)
    return ' '.join(processed_words)

df = pd.read_csv('movie_review.csv')
df.head()

X = df['text']
Y = df['tag']

df['text'] = df['text'].apply(preprocess_text)

print(X[0], Y[0])

# reduce overfitting
vect = CountVectorizer(max_features=2500)

X = vect.fit_transform(X)

print(vect.inverse_transform(X[0]))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# with Native Bayes Bernulli model
model = BernoulliNB()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
Y_pred_train = model.predict(X_train)

acc = accuracy_score(Y_test, Y_pred)
acc_train = accuracy_score(Y_train, Y_pred_train)

print(f'TEST - Acc: {acc}')
print(f'TRAIN - Acc: {acc_train}')

# trying different Models
models = [BernoulliNB(), MultinomialNB(), LogisticRegression(max_iter=500), MLPClassifier(hidden_layer_sizes=(2), verbose=False)]

for model in models:
    model = model
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    Y_pred_train = model.predict(X_train)

    acc = accuracy_score(Y_test, Y_pred)
    acc_train = accuracy_score(Y_train, Y_pred_train)

    print('=' * 50)
    print(f'Model: {model}' )
    print(f'TEST - Acc: {acc}')
    print(f'TRAIN - Acc: {acc_train}')
    print('=' * 50)

# linear support vector machine
svc = LinearSVC(dual='auto')
svc.fit(X_train, Y_train)

acc = svc.score(X_test, Y_test)
acc_train = svc.score(X_train, Y_train)

print(f'SVM - TEST: {acc} / TRAIN: {acc_train}')

# saving model and vect
dump(svc, 'sentiment_model.joblib')
dump(vect, 'count_vectorizer.joblib')